import asyncio
import csv
import functools
import json
import os
import random
import re
import sys
from pathlib import Path

import pymorphy3
from anthropic import APIConnectionError, APIError, APIStatusError
from dotenv import load_dotenv

from llm_backends import OllamaError

load_dotenv()

morph = pymorphy3.MorphAnalyzer()
DATA_DIR = Path(__file__).parent / "data"


def _log(message: str) -> None:
    """Печать, устойчивая к кодировке консоли.

    Сообщения об ошибках приходят из API и могут содержать символы,
    которых нет в cp1251 (например, полноширинный ＄). Обычный print()
    на такой строке падает с UnicodeEncodeError — то есть обработчик
    ошибки сам роняет процесс.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(message.encode(encoding, errors="replace").decode(encoding) + "\n")

# Стоп-слова (служебные, очень частые)
STOP_WORDS = {
    'и', 'в', 'на', 'с', 'к', 'по', 'за', 'из', 'о', 'у', 'от', 'для', 'при',
    'до', 'или', 'что', 'как', 'это', 'который', 'его', 'её', 'их', 'свой',
    'этот', 'тот', 'весь', 'сам', 'быть', 'а', 'но', 'да', 'не', 'ни', 'же',
    'бы', 'ли', 'вот', 'только', 'ещё', 'уже', 'где', 'когда', 'если', 'чтобы'
}

# ── Эвристики для отсева артефактов ──────────────────────────────────────
MIN_WORD_LEN = 3
MAX_WORD_LEN = 30
MIN_VOWEL_RATIO = 0.25
VOWELS = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')

# Запрещённые сочетания согласных (характерны для артефактов)
INVALID_CONSONANT_CLUSTERS = [
    'пкн', 'тпт', 'рьщ', 'гнн', 'бщт', 'вщр',
]


class WordListManager:
    """Управление списками слов по классам и словарём Шарова (оптимизировано, без pandas)"""

    def __init__(self):
        self.freq_lists = {}
        self.relative_lists = {}
        self.sharov = {}
        self._load_all()

    def _load_all(self):
        self._load_class_lists()
        self._load_sharov()

    def _load_class_lists(self):
        for class_num in range(2, 12):
            # Частотный список
            freq_path = DATA_DIR / f"class_{class_num}_freq.csv"
            if freq_path.exists():
                self.freq_lists[class_num] = {}
                try:
                    with open(freq_path, encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if not row:
                                continue
                            word = row[0].strip().lower()
                            # Пропускаем пустые строки, NaN и возможные заголовки
                            if not word or word == 'nan' or word == 'word' or word == 'слово':
                                continue

                            freq = 1.0
                            if len(row) > 1 and row[1].strip():
                                try:
                                    freq = float(row[1].strip())
                                except ValueError:
                                    pass  # Если не получилось преобразовать в число, оставляем 1.0

                            self.freq_lists[class_num][word] = freq
                    print(f"[OK] Freq class {class_num}: {len(self.freq_lists[class_num])} words")
                except Exception as e:
                    print(f"[ERROR] Freq class {class_num}: {e}")

            # Относительный список
            rel_path = DATA_DIR / f"class_{class_num}_relative.csv"
            if rel_path.exists():
                self.relative_lists[class_num] = set()
                try:
                    with open(rel_path, encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if not row:
                                continue
                            word = row[0].strip().lower()
                            if not word or word == 'nan' or word == 'word' or word == 'слово':
                                continue

                            self.relative_lists[class_num].add(word)
                    print(f"[OK] Relative class {class_num}: {len(self.relative_lists[class_num])} words")
                except Exception as e:
                    print(f"[ERROR] Relative class {class_num}: {e}")

    def _load_sharov(self):
        sharov_path = DATA_DIR / "sharov.csv"
        if not sharov_path.exists():
            return

        # Формат файла: Lemma <TAB> PoS <TAB> Freq(ipm) <TAB> R <TAB> D <TAB> Doc
        # Колонку частоты ищем по заголовку, а не по фиксированному индексу:
        # раньше здесь читался parts[1] — то есть часть речи ('conj', 's', 'v').
        # float() от неё всегда бросал ValueError, ValueError гасился continue,
        # и словарь на 52 139 строк молча оставался пустым.
        def _split(line: str) -> list:
            for sep in ('\t', ';', ','):
                if sep in line:
                    return line.split(sep)
            return line.split()

        try:
            freq_idx = 2  # значение по умолчанию для известного формата
            with open(sharov_path, encoding='utf-8') as f:
                for lineno, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    parts = _split(line)

                    if lineno == 0:
                        header = [p.strip().lower() for p in parts]
                        for i, name in enumerate(header):
                            if 'freq' in name or 'ipm' in name:
                                freq_idx = i
                                break
                        if header and header[0] in ('lemma', 'word', 'слово'):
                            continue  # это заголовок, не данные

                    if len(parts) <= freq_idx:
                        continue
                    word = parts[0].strip().lower()
                    if not word or word in ('nan', 'word', 'lemma'):
                        continue
                    try:
                        freq = float(parts[freq_idx].strip())
                    except ValueError:
                        continue
                    # Лемма встречается по разу на каждую часть речи —
                    # суммируем, нас интересует употребительность слова в целом.
                    self.sharov[word] = self.sharov.get(word, 0.0) + freq
            print(f"[OK] Sharov: {len(self.sharov)} words")
        except OSError as e:
            print(f"[ERROR] Sharov: {e}")

    def get_word_frequency_in_class(self, word: str, class_num: int) -> float:
        word = word.lower().strip()
        if class_num in self.freq_lists:
            return self.freq_lists[class_num].get(word, 0)
        return 0

    def get_total_frequency_below_class(self, word: str, target_class: int) -> float:
        word = word.lower().strip()
        total = 0
        for class_num in range(2, target_class):
            if class_num in self.freq_lists:
                total += self.freq_lists[class_num].get(word, 0)
        return total

    def word_first_appears_in_class(self, word: str) -> int | None:
        word = word.lower().strip()
        for class_num in range(2, 12):
            if class_num in self.relative_lists:
                if word in self.relative_lists[class_num]:
                    return class_num
        return None

    def get_sharov_frequency(self, word: str) -> float:
        return self.sharov.get(word.lower().strip(), 0)

    def is_word_known_below_class(self, word: str, target_class: int) -> bool:
        first_class = self.word_first_appears_in_class(word)
        if first_class is not None and first_class < target_class:
            return True
        return self.get_sharov_frequency(word) >= 50

    def get_words_for_class(self, class_num: int) -> list[str]:
        if class_num in self.relative_lists:
            return list(self.relative_lists[class_num])
        return []


# По умолчанию рядом со словарями: вердикты — дорогие производные данные,
# их разумно версионировать вместе с корпусом, как lock-файл.
VERDICTS_PATH = Path(os.getenv("RULEX_VERDICTS_PATH") or DATA_DIR / "word_verdicts.tsv")
_VERDICTS_HEADER = (
    "# Вердикты о реальности слов корпуса, посчитанные офлайн\n"
    "# (scripts/prefilter_corpus.py). Колонки: слово, вердикт, источник.\n"
    "# Вердикт о слове не меняется от запроса к запросу, поэтому считается\n"
    "# один раз, а не при каждой генерации.\n"
)


@functools.lru_cache(maxsize=1)
def get_verdicts() -> dict[str, str]:
    """Сохранённые вердикты: слово -> "real" | "artifact".

    Кэшируется на процесс; новые вердикты становятся видны после
    перезапуска — для офлайн-предфильтрации этого достаточно.
    """
    verdicts: dict[str, str] = {}
    if not VERDICTS_PATH.exists():
        return verdicts
    for line in VERDICTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] in ("real", "artifact"):
            verdicts[parts[0]] = parts[1]
    return verdicts


def append_verdicts(rows: list[tuple[str, str, str]]) -> None:
    """Дописать вердикты (слово, вердикт, источник) в хранилище."""
    if not rows:
        return
    new_file = not VERDICTS_PATH.exists()
    with open(VERDICTS_PATH, "a", encoding="utf-8") as f:
        if new_file:
            f.write(_VERDICTS_HEADER)
        for word, verdict, source in rows:
            f.write(f"{word}\t{verdict}\t{source}\n")
    get_verdicts.cache_clear()


@functools.lru_cache(maxsize=1)
def get_client():
    """Один клиент на процесс — иначе теряется пул соединений.

    Какой именно бэкенд, решают переменные окружения (см. llm_backends):
    облачный Anthropic, совместимый шлюз через ANTHROPIC_BASE_URL или
    локальная модель через Ollama. Интерфейс у них одинаковый, поэтому
    остальной код о разнице не знает.
    """
    from llm_backends import build_client

    return build_client()


@functools.lru_cache(maxsize=1)
def get_word_manager() -> "WordListManager":
    """Единственный экземпляр словарей на процесс.

    Построение читает ~12.5 МБ CSV и занимает около 0.9 с. Раньше оно
    происходило на каждый запрос к четырём эндпоинтам, причём синхронно
    внутри async-обработчика — то есть блокировало весь event loop.
    """
    return WordListManager()


class QuestionGenerator:
    """Генератор вопросов для теста на словарный запас"""

    # Контекст для LLM о назначении теста
    SYSTEM_CONTEXT = """Ты помогаешь создавать тестовые задания для проверки СЛОВАРНОГО ЗАПАСА школьников.

Цель теста — проверить, знает ли ученик ЗНАЧЕНИЕ слова, а не специальные знания.

ВАЖНЫЕ ПРАВИЛА:
1. НЕ использовать территориальные слова (названия городов, регионов, стран)
2. НЕ использовать этнонимы (названия народов, национальностей)
3. НЕ использовать узкоспециальные термины (медицинские, юридические, технические)
4. НЕ использовать имена собственные
5. НЕ использовать региональные/диалектные слова
6. Слова должны быть общеупотребительными в русском языке
7. Значение слова должно быть понятно из общего образования, а не из специальных знаний"""

    # Схемы ответов. strict: true — вход инструмента валидируется по схеме,
    # поэтому для strict нужны additionalProperties: false и required.
    TOOL_REAL_WORDS = {
        "name": "report_real_words",
        "description": "Сообщить, какие слова из списка реально существуют в русском языке.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "real_words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Реально существующие слова из списка, в исходном написании",
                },
            },
            "required": ["real_words"],
            "additionalProperties": False,
        },
    }

    TOOL_SUITABILITY = {
        "name": "report_suitability",
        "description": "Сообщить, подходит ли слово для теста на словарный запас школьника.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "suitable": {"type": "boolean"},
                "reason": {"type": "string", "description": "Краткое объяснение"},
            },
            "required": ["suitable", "reason"],
            "additionalProperties": False,
        },
    }

    TOOL_DISTRACTORS = {
        "name": "report_distractors",
        "description": "Сообщить слова-дистракторы для тестового задания.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "distractors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Десять разных слов, по одному слову в элементе",
                },
            },
            "required": ["distractors"],
            "additionalProperties": False,
        },
    }

    def __init__(self):
        # Клиент и словари общие на процесс, состояние генерации — своё:
        # generation_log и filter_failures изменяемые, и общий экземпляр
        # генератора гонялся бы между параллельными запросами.
        self.client = get_client()
        # Модель вынесена в окружение: прежний claude-sonnet-4-20250514
        # снят с обслуживания и отвечал 404 на каждый вызов.
        self.model = os.getenv("RULEX_MODEL_GENERATION", "claude-sonnet-5")
        self.word_manager = get_word_manager()
        self.generation_log = []
        # Батчи, которые не удалось проверить из-за сбоя API.
        # Непустой список означает, что фильтрация прошла не полностью.
        self.filter_failures: list[dict] = []
        # Сколько раз ответ пришёл структурой, а сколько раз пришлось
        # откатиться на разбор текста. Совместимые шлюзы умеют молча
        # выбрасывать параметры запроса, и без счётчика этого не видно.
        self.structured_stats = {"tool": 0, "fallback": 0}

    def _log(self, step: str, data: dict):
        self.generation_log.append({"step": step, **data})

    def _get_pos(self, word: str) -> str | None:
        parsed = morph.parse(word)
        if not parsed:
            return None
        return parsed[0].tag.POS

    def _get_lemma(self, word: str) -> str:
        parsed = morph.parse(word)
        if parsed:
            return parsed[0].normal_form
        return word.lower()

    def _call_llm(self, prompt: str, max_tokens: int = 500, effort: str | None = None) -> str:
        """Один вызов модели с извлечением текстового ответа.

        Ответ может состоять из нескольких блоков, и текстовый — не всегда
        первый: у современных моделей рассуждение включено по умолчанию,
        поэтому content[0] нередко оказывается ThinkingBlock. Прежний код
        брал content[0].text вслепую и падал с AttributeError на 11 словах
        из 19 — недетерминированно, в зависимости от того, выдала ли модель
        блок рассуждения.

        effort ограничивает глубину рассуждения: для короткой классификации
        она не нужна, а p95 латентности из-за неё доходила до 98 секунд.
        """
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": self.SYSTEM_CONTEXT,
            "messages": [{"role": "user", "content": prompt}],
        }
        if effort:
            params["output_config"] = {"effort": effort}

        response = self.client.messages.create(**params)

        parts = [
            block.text for block in response.content
            if getattr(block, "type", None) == "text" and hasattr(block, "text")
        ]
        if not parts:
            raise ValueError(
                f"Модель не вернула текстового блока "
                f"(stop_reason={getattr(response, 'stop_reason', '?')}, "
                f"блоки: {[getattr(b, 'type', '?') for b in response.content]})"
            )
        return "\n".join(parts).strip()

    def _call_llm_structured(
        self, prompt: str, tool: dict, max_tokens: int = 1024, effort: str | None = None
    ) -> tuple[dict | None, str]:
        """Вызов с ответом по JSON-схеме через строгий tool use.

        Возвращает (данные, текст). Данные — вход инструмента, провалидированный
        по схеме (`strict: true`); если модель вместо вызова ответила текстом,
        данные None, а текст уходит в разбор-запаску.

        Почему инструмент, а не output_config.format. Замер на шлюзе
        router.cheap: format молча выбрасывается — запрос проходит без ошибки,
        а модель отвечает markdown-списком с пояснениями. Инструменты шлюз
        пропускает.

        Почему tool_choice auto, а не принудительный. У Sonnet 5 и Opus 5
        рассуждение включено по умолчанию, и принудительный выбор инструмента
        с ним несовместим — API отвечает 400. С auto и явной инструкцией
        в промпте все три облачные модели вызывали инструмент в 3 случаях
        из 3; для редкого отказа есть запаска.
        """
        params = self._structured_params(prompt, tool, max_tokens, effort)
        response = self.client.messages.create(**params)
        return self._parse_structured(response.content, tool)

    def _structured_params(
        self, prompt: str, tool: dict, max_tokens: int = 1024, effort: str | None = None
    ) -> dict:
        """Параметры структурного запроса.

        Вынесены отдельно, чтобы обычный вызов и запрос в Batches API
        собирались одинаково: иначе батч мерил бы не то же самое, что
        живой путь.
        """
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": self.SYSTEM_CONTEXT,
            "messages": [{
                "role": "user",
                "content": f"{prompt}\n\nОтветь, вызвав инструмент {tool['name']}.",
            }],
            "tools": [tool],
            "tool_choice": {"type": "auto"},
        }
        if effort:
            params["output_config"] = {"effort": effort}
        return params

    def _parse_structured(self, content, tool: dict) -> tuple[dict | None, str]:
        """(данные инструмента, текст-запаска) из блоков ответа."""
        for block in content or []:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool["name"]:
                self.structured_stats["tool"] += 1
                return dict(block.input), ""

        self.structured_stats["fallback"] += 1
        text = "\n".join(
            block.text for block in content or []
            if getattr(block, "type", None) == "text" and hasattr(block, "text")
        )
        return None, text.strip()

    @staticmethod
    def _clean_token(token: str) -> str:
        """Слово из текстового ответа без markdown и пунктуации.

        Модели оформляют списки жирным (`**стол**`) и кодом (`` `стол` ``).
        Прежний разбор снимал только точки и запятые, и выделенное слово
        не совпадало с исходным — реальное слово тихо считалось отвергнутым.
        """
        token = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", token)
        return token.strip().strip("*_`'\"«».,;:()[]").strip().lower()

    def _is_artifact(self, word: str) -> tuple[bool, str]:
        """
        Эвристическая проверка: является ли слово артефактом (мусором из датасета).
        Работает без LLM, поэтому очень быстрая.
        """
        w = word.lower().strip()

        # Слишком короткое
        if len(w) < MIN_WORD_LEN:
            return True, f"Слишком короткое ({len(w)} букв)"

        # Слишком длинное
        if len(w) > MAX_WORD_LEN:
            return True, f"Слишком длинное ({len(w)} букв)"

        # Недостаточно гласных — признак артефакта (пкно, тпт, рьщ...)
        vowel_count = sum(1 for c in w if c in VOWELS)
        vowel_ratio = vowel_count / len(w) if len(w) > 0 else 0
        if vowel_ratio < MIN_VOWEL_RATIO:
            return True, f"Мало гласных ({vowel_ratio:.0%}), вероятно артефакт"

        # Запрещённые кластеры согласных
        for cluster in INVALID_CONSONANT_CLUSTERS:
            if cluster in w:
                return True, f"Недопустимое сочетание букв '{cluster}'"

        # pymorphy3 не знает слово совсем (score < 0.05 означает полную неизвестность)
        parsed = morph.parse(w)
        if parsed:
            best = parsed[0]
            if best.score < 0.05:
                return True, f"Слово неизвестно морфологическому словарю (score={best.score:.3f})"

        return False, "OK"

    def _real_words_prompt(self, batch: list[str]) -> str:
        """Промпт проверки реальности для одного батча слов.

        Отдельный метод, чтобы живой фильтр и офлайн-предфильтрация
        корпуса (scripts/prefilter_corpus.py) спрашивали модель одинаково.
        """
        words_str = ', '.join(batch)
        return f"""Ты эксперт русского языка и лексикограф.

Из списка слов выбери ТОЛЬКО те, которые реально существуют в стандартном русском языке и есть в словарях (Ожегов, РАС, Викисловарь).

ОТСЕИВАЙ слова которые:
- Выдуманы или являются артефактами датасета
- Звучат похоже на реальные, но НЕ существуют в словарях
- Являются ошибочными или искажёнными формами реальных слов

Примеры ВЫМЫШЛЕННЫХ слов которые нужно отсеять:
- травие (звучит как "трава" но это НЕ слово)
- восьмибрат (выдуманное)
- плэда (неправильное написание)
- будрить (похоже на "будить" но это НЕ слово)
- сугибнуть (выдуманное)
- ведрик (похоже на уменьшительное от "ведро" но такого слова НЕТ)
- пкно, тпеть, рьщарить (явный мусор)

Список для проверки:
{words_str}

В ответ включи ТОЛЬКО реально существующие слова из этого списка, в том же написании."""

    def _real_words_params(self, batch: list[str]) -> dict:
        """Параметры запроса проверки реальности — и для живого вызова,
        и для Batches API."""
        # 30 слов заметно длиннее прежних 200 токенов: ответ
        # обрезался, и валидные слова молча терялись.
        return self._structured_params(
            self._real_words_prompt(batch), self.TOOL_REAL_WORDS,
            max_tokens=1024, effort="low")

    def _real_words_from_answer(self, batch: list[str], content) -> list[str]:
        """Какие слова батча модель подтвердила — по блокам её ответа."""
        data, text = self._parse_structured(content, self.TOOL_REAL_WORDS)
        if data is not None:
            confirmed = [self._clean_token(w) for w in data.get("real_words", [])]
        else:
            # Запаска на случай, когда модель ответила текстом
            confirmed = [self._clean_token(w) for w in re.split(r"[,\n;]+", text) if w.strip()]
        confirmed_set = {w for w in confirmed if w}
        # Оставляем только те, что были в батче И подтверждены: модель
        # иногда «исправляет» слово, и исправленного в батче нет.
        return [w for w in batch if w.lower() in confirmed_set]

    def _filter_real_words_batch(self, words: list[str], batch_size: int = 30) -> list[str]:
        """
        Проверяет сразу пачку слов одним вызовом LLM.
        Отсеивает выдуманные слова типа 'травие', 'восьмибрат', 'плэда'.

        При сбое API батч ОТБРАСЫВАЕТСЯ, а не пропускается целиком:
        лучше потерять слова, чем тихо пустить мусор в тест.
        Факт деградации виден в self.filter_failures.
        """
        real_words = []
        self.filter_failures = []
        total_batches = (len(words) + batch_size - 1) // batch_size

        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            try:
                response = self.client.messages.create(**self._real_words_params(batch))
                valid_in_batch = self._real_words_from_answer(batch, response.content)
                rejected = [w for w in batch if w not in valid_in_batch]
                if rejected:
                    _log(f"[FAKE] Отсеяно {len(rejected)} вымышленных слов: {', '.join(rejected[:5])}")
                real_words.extend(valid_in_batch)

            except (APIStatusError, APIConnectionError, APIError,
                    OllamaError) as e:
                # Раньше здесь в результат добавлялся ВЕСЬ батч, включая мусор:
                # сбой API молча отключал фильтрацию, и снаружи это было не видно.
                # Теперь батч отбрасывается, а факт деградации фиксируется.
                self.filter_failures.append({
                    "batch_size": len(batch),
                    "error": f"{type(e).__name__}: {e}",
                })
                _log(f"[ERROR] Батч из {len(batch)} слов отброшен: {type(e).__name__}")

        if self.filter_failures:
            _log(f"[WARN] Фильтр деградировал: {len(self.filter_failures)} "
                 f"батч(ей) из {total_batches} не проверено и отброшено")
        return real_words

    def _is_dictionary_word(self, word: str) -> bool:
        """Слово подтверждено словарём — обращаться к модели не нужно.

        Два независимых источника: словарь OpenCorpora внутри pymorphy3
        (`word_is_known` — именно в словаре, а не угадано по аналогии)
        и частотный словарь Шарова на 51 682 леммы.
        """
        return (
            morph.word_is_known(word)
            or self.word_manager.get_sharov_frequency(word) > 0
        )

    def _confirm_real(self, words: list[str]) -> list[str]:
        """Отбор реальных слов каскадом: сохранённые вердикты, словари, модель.

        1. Вердикты, посчитанные офлайн (scripts/prefilter_corpus.py): вердикт
           о слове не меняется от запроса к запросу, и пересчитывать его
           при каждой генерации незачем.
        2. Словари: подтверждают 61% корпуса (от 92% во 2 классе до 37%
           в 11). На 220 размеченных словах каскад словари -> модель дал
           F1 0.86 против 0.78 у словарей и 0.75 у модели по отдельности.
        3. Модель — только для того, что не решено первыми двумя шагами.
        """
        verdicts = get_verdicts()
        stored_real = [w for w in words if verdicts.get(w) == "real"]
        undecided = [w for w in words if w not in verdicts]

        confirmed = stored_real + [w for w in undecided if self._is_dictionary_word(w)]
        unknown = [w for w in undecided if not self._is_dictionary_word(w)]
        _log(
            f"[INFO] Из {len(words)}: вердикт сохранён у {len(words) - len(undecided)}, "
            f"словари подтвердили {len(confirmed) - len(stored_real)}, "
            f"через модель пойдут {len(unknown)}"
        )
        if not unknown:
            return confirmed
        return confirmed + self._filter_real_words_batch(unknown, batch_size=30)

    def _check_word_suitability(self, word: str) -> tuple[bool, str]:
        """Проверка слова через LLM на пригодность для теста словарного запаса"""

        prompt = f"""Оцени, подходит ли слово "{word}" для теста на СЛОВАРНЫЙ ЗАПАС школьника.

Слово НЕ подходит, если это:
- Название места (город, страна, регион, река, гора)
- Название народа или национальности (латыш, немец, татарин)
- Прилагательное от географического названия (московский, тверской, балтийский)
- Узкоспециальный термин (медицинский, юридический, технический)
- Устаревшее или диалектное слово
- Имя собственное
- Слово, требующее специальных знаний для понимания

Слово ПОДХОДИТ, если это общеупотребительное слово, значение которого можно объяснить без специальных знаний."""

        try:
            # Прежний разбор: `"ПОДХОДИТ: да" in response.lower()` — левая часть
            # никогда не срабатывала (верхний регистр ищется в нижнем), а на
            # ответе «Подходит — да» не срабатывала и правая.
            data, text = self._call_llm_structured(
                prompt, self.TOOL_SUITABILITY, max_tokens=512, effort="low")
        except (APIStatusError, APIConnectionError, APIError, OllamaError) as e:
            # Раньше здесь возвращалось (True, ...): сбой API объявлял слово
            # пригодным и молча пропускал в тест топонимы и узкие термины.
            # Та же ошибка, что была в батч-фильтре; закрываемся так же.
            self._log("word_check_error", {"word": word, "error": str(e)})
            return False, f"Не удалось проверить: {type(e).__name__}"

        if data is not None:
            is_suitable = bool(data.get("suitable"))
            reason = str(data.get("reason") or ("OK" if is_suitable else "Не подходит"))
        else:
            # Запаска: ищем явное «да»/«нет» в тексте
            lowered = text.lower()
            is_suitable = bool(re.search(r"подходит\W{0,5}да\b", lowered)) and "не подходит" not in lowered
            reason = text[:200] or "Не удалось разобрать ответ"

        self._log("word_check", {"word": word, "suitable": is_suitable, "reason": reason})
        return is_suitable, reason

    def _is_basic_valid(self, word: str) -> tuple[bool, str]:
        """Базовая проверка слова (без LLM)"""
        word = word.lower().strip()

        if not word or len(word) < 2:
            return False, "Слишком короткое"

        if not word.isalpha():
            return False, "Содержит не-буквы"

        if '-' in word:
            return False, "Содержит дефис"

        pos = self._get_pos(word)
        if pos not in ['NOUN', 'VERB', 'INFN']:
            return False, f"Неподходящая часть речи: {pos}"

        parsed = morph.parse(word)
        if parsed:
            tags = str(parsed[0].tag)
            if 'Abbr' in tags or 'NUMB' in tags:
                return False, "Аббревиатура или число"

        return True, "OK"

    def _get_distractors(self, word: str, target_class: int) -> list[str]:
        """Получение дистракторов"""
        pos = self._get_pos(word)
        pos_rus = {'NOUN': 'существительное', 'VERB': 'глагол', 'INFN': 'инфинитив'}.get(pos, 'существительное')
        example = ("дом, лес, река, гора, поле, берег, холм, овраг, поляна, роща"
                   if pos == 'NOUN'
                   else "бежать, идти, прыгать, ползти, лететь, плыть, ехать, спешить, брести, мчаться")

        prompt = f"""Для теста на словарный запас нужны слова-дистракторы к слову "{word}" ({pos_rus}).

Требования к дистракторам:
1. Должны быть гиперонимами или гипонимами слова "{word}"
2. НЕ синонимы слова "{word}" (значение должно быть ДРУГИМ)
3. НЕ однокоренные со словом "{word}"
4. Та же часть речи ({pos_rus})
5. Общеупотребительные слова (не специальные термины)
6. НЕ географические названия, НЕ этнонимы
7. Одно слово каждое, без дефисов
8. Все десять слов РАЗНЫЕ, повторы недопустимы

Дай 10 слов. Не обсуждай само слово "{word}" и не оценивай запрос.

Пример подходящего набора для другого слова:
{example}"""

        data, text = self._call_llm_structured(prompt, self.TOOL_DISTRACTORS, max_tokens=1024)
        self._log("distractors_response", {"word": word, "structured": data is not None,
                                           "response": data if data is not None else text})

        if data is not None:
            return self._select_distractors(data.get("distractors", []), word)
        return self._parse_distractors(text, word)

    @staticmethod
    def _extract_list_line(response: str) -> str:
        """Строка со списком из ответа модели.

        Прежний код брал всё после первого двоеточия. На модели, которая
        вместо списка пишет рассуждение («такого слова не существует...»),
        это попадало в середину прозы и не давало ни одного кандидата.
        Берём самую «списочную» строку: с запятыми и без длинных слов.
        """
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        # Нумерованный или маркированный список -> склеиваем в одну строку
        bullets = [re.sub(r'^[\s\-\*•]*\d*[.)]?\s*', '', ln)
                   for ln in lines if re.match(r'^[\s\-\*•]*\d*[.)]?\s*\S+$', ln)]
        if len(bullets) >= 3 and all(len(b.split()) <= 2 for b in bullets):
            return ", ".join(bullets)

        # Иначе — строка с наибольшим числом запятых
        best = max(lines, key=lambda ln: ln.count(","), default="")
        if best.count(",") >= 2:
            # Отрезаем возможную преамбулу вида «Вот слова: a, b, c»
            return best.split(":", 1)[-1] if ":" in best else best
        return response

    def _parse_distractors(self, response: str, word: str) -> list[str]:
        """Разбор ответа в список дистракторов.

        Отличия от прежнего разбора: снимается нумерация, отсеиваются
        повторы (модели охотно выдают одно слово по три раза — у qwen3
        это давало 21% заданий с двумя одинаковыми вариантами) и часть
        речи проверяется по ВСЕМ разборам pymorphy3, а не только по
        самому вероятному: «род» как существительное иначе теряется,
        потому что первым разбором идёт глагольная форма.
        """
        text = self._extract_list_line(response)
        return self._select_distractors(re.split(r"[,;\n]+", text), word)

    def _select_distractors(self, candidates: list[str], word: str) -> list[str]:
        """Отбор дистракторов из кандидатов — общий для структурного ответа
        и для разбора текста.

        Схема гарантирует форму ответа (список строк), но не лингвистику:
        повторы, другая часть речи и однокоренные с целевым словом
        приходят и в валидном JSON, поэтому проверки те же.
        """
        target_pos = self._get_pos(word)
        seen_lemmas = {self._get_lemma(word)}
        distractors = []
        for candidate in candidates:
            candidate = self._clean_token(str(candidate))
            if not candidate or len(candidate) < 2 or not candidate.isalpha():
                continue
            if candidate == word.lower():
                continue
            if not self._pos_matches(candidate, target_pos):
                continue

            lemma = self._get_lemma(candidate)
            if lemma in seen_lemmas:
                continue  # и дубль, и однокоренное с целевым словом
            seen_lemmas.add(lemma)

            distractors.append(candidate)
            if len(distractors) >= 3:
                break

        return distractors

    @staticmethod
    def _pos_matches(candidate: str, target_pos: str | None) -> bool:
        """Совпадает ли часть речи хотя бы по одному разбору.

        pymorphy3 возвращает разборы по убыванию вероятности, и у
        омонимов верный не всегда первый: «род» разбирается сначала как
        глагольная форма, и нормальное существительное отбрасывалось.
        """
        if target_pos is None:
            return False
        for parse in morph.parse(candidate):
            pos = parse.tag.POS
            if pos == target_pos:
                return True
            # INFN и VERB — одна часть речи для целей теста
            if {pos, target_pos} <= {'VERB', 'INFN'}:
                return True
        return False

    def _get_definition(self, word: str, distractors: list[str], target_class: int) -> str:
        """Получение толкования"""
        pos = self._get_pos(word)
        pos_rus = {'NOUN': 'существительное', 'VERB': 'глагол', 'INFN': 'инфинитив'}.get(pos, 'существительное')

        forbidden = ', '.join([word] + distractors)

        prompt = f"""Напиши толкование для слова "{word}" ({pos_rus}) для теста на словарный запас школьника.

ВАЖНО:
1. Толкование должно быть кратким и понятным
2. НЕ используй слова: {forbidden} и однокоренные им
3. НЕ используй специальные термины
4. НЕ ссылайся на географические названия или национальности
5. Используй только простые, общеупотребительные слова
6. Толкование должно однозначно указывать на слово "{word}"

Напиши только само толкование, без целевого слова и тире:"""

        definition = self._call_llm(prompt, max_tokens=150)
        definition = definition.strip().lstrip('-—').strip()

        self._log("definition_initial", {"word": word, "definition": definition})

        # Проверка и корректировка
        for attempt in range(3):
            # Проверяем на наличие запрещённых слов
            needs_correction = False

            words_in_def = re.findall(r'[а-яёА-ЯЁ]+', definition.lower())
            for w in words_in_def:
                lemma = self._get_lemma(w)
                # Проверяем однокоренные
                for forbidden_word in [word] + distractors:
                    if lemma == self._get_lemma(forbidden_word):
                        needs_correction = True
                        break
                    # Простая проверка на общий корень
                    if len(lemma) > 3 and len(self._get_lemma(forbidden_word)) > 3:
                        if lemma[:4] == self._get_lemma(forbidden_word)[:4]:
                            needs_correction = True
                            break

            if not needs_correction:
                break

            # Корректировка
            correction_prompt = f"""Исправь толкование так, чтобы не использовать слова, однокоренные с: {forbidden}

Текущее толкование: "{definition}"

Сохрани смысл, но перефразируй. Напиши только исправленное толкование:"""

            definition = self._call_llm(correction_prompt, max_tokens=150)
            definition = definition.strip().lstrip('-—').strip()
            self._log("definition_corrected", {"attempt": attempt + 1, "definition": definition})

        self._validate_definition(definition, word)
        return definition

    # Признаки того, что модель не дала толкование, а отказалась или
    # стала рассуждать о запросе. Проверяется только начало ответа.
    _REFUSAL_MARKERS = (
        "я не могу", "не могу выполнить", "не могу создать", "к сожалению",
        "не подходит для теста", "это узкоспециальн", "как языковая модель",
    )

    def _validate_definition(self, definition: str, word: str) -> None:
        """Последний рубеж: толкование, по которому ответ очевиден, не сохраняется.

        Раньше, исчерпав три попытки исправления, код возвращал последний
        вариант как есть — даже если в нём по-прежнему стояло загаданное
        слово. Так в банк вопросов попадали отказы модели: «Я не могу
        создать корректное толкование для слова "микробарограф"…» сохранялось
        как толкование, и ответ был написан прямо в задании.

        Здесь критерий строгий и без ложных срабатываний: совпадение леммы
        с загаданным словом, а не сравнение по первым буквам, как в цикле
        исправлений.
        """
        head = definition.lower()[:120]
        if any(marker in head for marker in self._REFUSAL_MARKERS):
            raise ValueError(f"Модель отказалась толковать '{word}'")

        target = self._get_lemma(word)
        for token in re.findall(r"[а-яёА-ЯЁ]+", definition):
            if self._get_lemma(token.lower()) == target:
                raise ValueError(f"Толкование для '{word}' содержит само слово")

    def generate_question(self, word: str, word_class: int = 6, frequency_type: str = "medium") -> dict:
        """Генерация одного вопроса"""
        self.generation_log = []
        word = word.lower().strip()

        self._log("start", {"word": word, "word_class": word_class})

        # Базовая проверка
        is_valid, reason = self._is_basic_valid(word)
        if not is_valid:
            raise ValueError(f"Слово '{word}' не подходит: {reason}")

        # Получение дистракторов
        distractors = self._get_distractors(word, word_class)

        if len(distractors) < 2:
            raise ValueError(f"Не удалось найти дистракторы для '{word}'")

        # Получение толкования
        definition = self._get_definition(word, distractors, word_class)

        # Форматирование
        if definition and definition[0].islower():
            definition = definition[0].upper() + definition[1:]
        definition = definition.rstrip('.')
        if not definition.endswith('—') and not definition.endswith('-'):
            definition = definition + ' —'

        result = {
            "target_word": word,
            "definition": definition,
            "correct_answer": word,
            "distractors": distractors,
            "part_of_speech": self._get_pos(word),
            "word_class": word_class,
            "frequency_type": frequency_type,
            "generation_log": json.dumps(self.generation_log, ensure_ascii=False, default=str)
        }

        self._log("complete", {"success": True})
        return result

    def _prepare_candidates(self, word_class: int, count: int) -> tuple[list[str], list[str]]:
        """Шаги 1–4: эвристики, батч-проверка реальности, распределение частотности.

        Вынесено отдельно, чтобы синхронная и асинхронная генерация
        использовали ровно одну и ту же подготовку.
        """
        all_words = self.word_manager.get_words_for_class(word_class)

        if not all_words:
            raise ValueError(f"Нет списка слов для {word_class} класса")

        # ── Шаг 1: эвристическая фильтрация (бесплатно) ─────────────────
        clean_words = []
        artifact_count = 0

        for w in all_words:
            # Базовая проверка
            is_valid, _ = self._is_basic_valid(w)
            if not is_valid:
                artifact_count += 1
                continue

            # Эвристика артефактов
            is_art, _ = self._is_artifact(w)
            if is_art:
                artifact_count += 1
                continue

            clean_words.append(w)

        print(f"[INFO] Class {word_class}: {len(all_words)} всего, {artifact_count} артефактов удалено, {len(clean_words)} чистых слов")

        if len(clean_words) < 5:
            raise ValueError(
                f"Недостаточно валидных слов для {word_class} класса "
                f"(после фильтрации осталось {len(clean_words)})"
            )

        # ── Шаг 2: батч-проверка реальности через LLM ────────────────────
        random.shuffle(clean_words)

        # Берём пул с запасом для LLM-проверки (count * 5 = 100 слов)
        # Это займёт примерно 100/30 ≈ 4 вызова LLM вместо 100
        check_pool_size = min(len(clean_words), count * 5)
        check_pool = clean_words[:check_pool_size]

        real_words = self._confirm_real(check_pool)

        # ── Шаг 3: если мало — добираем из оставшихся ────────────────────
        if len(real_words) < count * 2:
            _log("[INFO] Мало реальных слов, проверяю дополнительный батч...")
            extra_pool = clean_words[check_pool_size : check_pool_size + count * 3]
            if extra_pool:
                real_words.extend(self._confirm_real(extra_pool))
                _log(f"[INFO] После дополнительной проверки: {len(real_words)} реальных слов")

        if len(real_words) < 5:
            raise ValueError(f"Критически мало реальных слов: {len(real_words)}")

        # ── Шаг 4: распределение частотности ─────────────────────────────
        freq_distribution = (
            ["high"] * int(count * 0.4) +
            ["medium"] * int(count * 0.4) +
            ["low"] * int(count * 0.2)
        )
        random.shuffle(freq_distribution)

        candidates = real_words[:min(len(real_words), count * 8)]
        return candidates, freq_distribution

    def generate_questions_for_class(self, word_class: int, count: int = 20) -> list[dict]:
        """
        Автоматическая генерация вопросов для класса.

        Изменения:
        1. Эвристическая фильтрация артефактов (быстро, без LLM)
        2. Батч-проверка реальности слов через LLM (экономия ~95% вызовов)
        3. Увеличенный пул кандидатов
        """

        candidates, freq_distribution = self._prepare_candidates(word_class, count)

        questions = []

        for word in candidates:
            if len(questions) >= count:
                break

            # Проверка пригодности для теста (специальные термины и т.д.)
            is_suitable, reason = self._check_word_suitability(word)
            if not is_suitable:
                print(f"[SKIP] {word}: {reason}")
                continue

            try:
                freq_type = (
                    freq_distribution[len(questions)]
                    if len(questions) < len(freq_distribution)
                    else "medium"
                )
                question = self.generate_question(word, word_class, freq_type)
                questions.append(question)
                print(f"[OK] {len(questions)}/{count}: {word}")
            except Exception as e:
                print(f"[ERROR] {word}: {e}")
                continue

        # ── Шаг 6: проверка результата ───────────────────────────────────
        if len(questions) < count:
            print(
                f"[WARN] Удалось сгенерировать только {len(questions)}/{count} вопросов. "
                f"Проверьте качество словаря class_{word_class}_relative.csv"
            )

        if len(questions) < max(5, count // 2):
            raise ValueError(
                f"Критически мало вопросов: {len(questions)}/{count}. "
                f"Словарь класса {word_class} содержит слишком много артефактов или специальных терминов."
            )

        return questions

    def _make_one(self, word: str, word_class: int, freq_type: str) -> dict | None:
        """Одно задание целиком. Своё состояние на вызов.

        Свой экземпляр генератора нужен потому, что generation_log
        изменяемый: при параллельном запуске задачи затирали бы логи
        друг друга. Клиент и словари при этом общие (см. lru_cache),
        так что создание экземпляра почти бесплатно.
        """
        worker = QuestionGenerator()
        suitable, reason = worker._check_word_suitability(word)
        if not suitable:
            _log(f"[SKIP] {word}: {reason}")
            return None
        try:
            return worker.generate_question(word, word_class, freq_type)
        except (ValueError, APIStatusError, APIConnectionError, APIError, OllamaError) as exc:
            _log(f"[ERROR] {word}: {type(exc).__name__}")
            return None

    async def agenerate_questions_for_class(
        self, word_class: int, count: int = 20, concurrency: int | None = None
    ) -> list[dict]:
        """Асинхронная генерация: вызовы уходят из event loop в поток.

        Главное здесь — НЕ ускорение, а то, что обработчик перестаёт
        блокировать сервер. Синхронная версия делает 3–6 вызовов LLM
        на слово подряд, и на время генерации (минуты) приложение
        не отвечало никому.

        Про параллелизм. По умолчанию он ВЫКЛЮЧЕН (concurrency=1),
        и это результат замера, а не осторожность: на шести словах
        пять параллельных задач дали через шлюз router.cheap 251 секунду
        против 53 последовательных, то есть впятеро хуже — запросы
        троттлятся и SDK уходит в ретраи. На локальной Ollama выигрыша
        тоже нет: она обслуживает один GPU и выполняет запросы по очереди.
        Повышать RULEX_GEN_CONCURRENCY имеет смысл только против прямого
        API и только с замером — вслепую это делает хуже.
        """
        if concurrency is None:
            concurrency = max(1, int(os.getenv("RULEX_GEN_CONCURRENCY") or 1))
        candidates, freq_distribution = await asyncio.to_thread(
            self._prepare_candidates, word_class, count
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def one(word: str, freq_type: str) -> dict | None:
            async with semaphore:
                return await asyncio.to_thread(self._make_one, word, word_class, freq_type)

        questions: list[dict] = []
        pending = list(candidates)
        while pending and len(questions) < count:
            wave, pending = pending[:count - len(questions)], pending[count - len(questions):]
            tasks = [
                one(word, freq_distribution[(len(questions) + i) % len(freq_distribution)]
                    if freq_distribution else "medium")
                for i, word in enumerate(wave)
            ]
            for result in await asyncio.gather(*tasks):
                if result is not None and len(questions) < count:
                    questions.append(result)
            _log(f"[INFO] Готово {len(questions)}/{count}")

        if len(questions) < max(5, count // 2):
            raise ValueError(
                f"Критически мало вопросов: {len(questions)}/{count}. "
                f"Словарь класса {word_class} содержит слишком много артефактов "
                f"или специальных терминов."
            )
        return questions

    def has_word_list(self, word_class: int) -> bool:
        return word_class in self.word_manager.relative_lists and len(self.word_manager.relative_lists[word_class]) > 0

    def get_available_classes(self) -> list[int]:
        return sorted(self.word_manager.relative_lists.keys())
