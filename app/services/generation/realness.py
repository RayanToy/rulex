"""Этап 1: реальное ли слово.

Списки слов по классам зашумлены: в них есть обрывки, искажённые формы
и слова, которых нет ни в одном словаре. Отсев идёт от дешёвого
к дорогому: эвристики, сохранённые вердикты, словари и только потом
модель.
"""
import re

from anthropic import APIConnectionError, APIError, APIStatusError

from app.services.generation.console import log
from app.services.generation.model_calls import ModelCaller, clean_token
from app.services.generation.morphology import get_pos, morph
from app.services.llm import OllamaError
from app.services.wordlists import get_verdicts

# ── Эвристики для отсева артефактов ──────────────────────────────────────
MIN_WORD_LEN = 3
MAX_WORD_LEN = 30
MIN_VOWEL_RATIO = 0.25
VOWELS = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')

# Запрещённые сочетания согласных (характерны для артефактов)
INVALID_CONSONANT_CLUSTERS = [
    'пкн', 'тпт', 'рьщ', 'гнн', 'бщт', 'вщр',
]

# Схема ответа. strict: true — вход инструмента валидируется по схеме,
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


def is_basic_valid(word: str) -> tuple[bool, str]:
    """Базовая проверка слова (без LLM)"""
    word = word.lower().strip()

    if not word or len(word) < 2:
        return False, "Слишком короткое"

    if not word.isalpha():
        return False, "Содержит не-буквы"

    if '-' in word:
        return False, "Содержит дефис"

    pos = get_pos(word)
    if pos not in ['NOUN', 'VERB', 'INFN']:
        return False, f"Неподходящая часть речи: {pos}"

    parsed = morph.parse(word)
    if parsed:
        tags = str(parsed[0].tag)
        if 'Abbr' in tags or 'NUMB' in tags:
            return False, "Аббревиатура или число"

    return True, "OK"


def is_artifact(word: str) -> tuple[bool, str]:
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


def real_words_prompt(batch: list[str]) -> str:
    """Промпт проверки реальности для одного батча слов.

    Общий для живого фильтра и офлайн-предфильтрации корпуса
    (scripts/prefilter_corpus.py): оба спрашивают модель одинаково.
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


class RealnessFilter:
    """Каскад проверки реальности: вердикты, словари, модель."""

    def __init__(self, llm: ModelCaller, word_manager):
        self.llm = llm
        self.word_manager = word_manager
        # Батчи, которые не удалось проверить из-за сбоя API.
        # Непустой список означает, что фильтрация прошла не полностью.
        self.failures: list[dict] = []

    def is_dictionary_word(self, word: str) -> bool:
        """Слово подтверждено словарём — обращаться к модели не нужно.

        Два независимых источника: словарь OpenCorpora внутри pymorphy3
        (`word_is_known` — именно в словаре, а не угадано по аналогии)
        и частотный словарь Шарова на 51 682 леммы.
        """
        return (
            morph.word_is_known(word)
            or self.word_manager.get_sharov_frequency(word) > 0
        )

    def confirm(self, words: list[str]) -> list[str]:
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

        confirmed = stored_real + [w for w in undecided if self.is_dictionary_word(w)]
        unknown = [w for w in undecided if not self.is_dictionary_word(w)]
        log(
            f"[INFO] Из {len(words)}: вердикт сохранён у {len(words) - len(undecided)}, "
            f"словари подтвердили {len(confirmed) - len(stored_real)}, "
            f"через модель пойдут {len(unknown)}"
        )
        if not unknown:
            return confirmed
        return confirmed + self.filter_batch(unknown, batch_size=30)

    def request_params(self, batch: list[str]) -> dict:
        """Параметры запроса проверки реальности — и для живого вызова,
        и для Batches API."""
        # 30 слов заметно длиннее прежних 200 токенов: ответ
        # обрезался, и валидные слова молча терялись.
        return self.llm.structured_params(
            real_words_prompt(batch), TOOL_REAL_WORDS, max_tokens=1024, effort="low")

    def words_from_answer(self, batch: list[str], content) -> list[str]:
        """Какие слова батча модель подтвердила — по блокам её ответа."""
        data, text = self.llm.parse_structured(content, TOOL_REAL_WORDS)
        if data is not None:
            confirmed = [clean_token(w) for w in data.get("real_words", [])]
        else:
            # Запаска на случай, когда модель ответила текстом
            confirmed = [clean_token(w) for w in re.split(r"[,\n;]+", text) if w.strip()]
        confirmed_set = {w for w in confirmed if w}
        # Оставляем только те, что были в батче И подтверждены: модель
        # иногда «исправляет» слово, и исправленного в батче нет.
        return [w for w in batch if w.lower() in confirmed_set]

    def filter_batch(self, words: list[str], batch_size: int = 30) -> list[str]:
        """
        Проверяет сразу пачку слов одним вызовом LLM.
        Отсеивает выдуманные слова типа 'травие', 'восьмибрат', 'плэда'.

        При сбое API батч ОТБРАСЫВАЕТСЯ, а не пропускается целиком:
        лучше потерять слова, чем тихо пустить мусор в тест.
        Факт деградации виден в self.failures.
        """
        real_words = []
        self.failures = []
        total_batches = (len(words) + batch_size - 1) // batch_size

        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            try:
                response = self.llm.client.messages.create(**self.request_params(batch))
                valid_in_batch = self.words_from_answer(batch, response.content)
                rejected = [w for w in batch if w not in valid_in_batch]
                if rejected:
                    log(f"[FAKE] Отсеяно {len(rejected)} вымышленных слов: {', '.join(rejected[:5])}")
                real_words.extend(valid_in_batch)

            except (APIStatusError, APIConnectionError, APIError,
                    OllamaError) as e:
                # Раньше здесь в результат добавлялся ВЕСЬ батч, включая мусор:
                # сбой API молча отключал фильтрацию, и снаружи это было не видно.
                # Теперь батч отбрасывается, а факт деградации фиксируется.
                self.failures.append({
                    "batch_size": len(batch),
                    "error": f"{type(e).__name__}: {e}",
                })
                log(f"[ERROR] Батч из {len(batch)} слов отброшен: {type(e).__name__}")

        if self.failures:
            log(f"[WARN] Фильтр деградировал: {len(self.failures)} "
                f"батч(ей) из {total_batches} не проверено и отброшено")
        return real_words
