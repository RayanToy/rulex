"""Словари корпуса и хранилище вердиктов о реальности слов.

Частотные списки по классам (data/class_N_freq.csv, class_N_relative.csv),
словарь Шарова и офлайн-вердикты из scripts/prefilter_corpus.py. Всё это
читается с диска один раз на процесс.
"""
import csv
import functools
import os
from pathlib import Path

from app import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"


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


@functools.lru_cache(maxsize=1)
def get_word_manager() -> "WordListManager":
    """Единственный экземпляр словарей на процесс.

    Построение читает ~12.5 МБ CSV и занимает около 0.9 с. Раньше оно
    происходило на каждый запрос к четырём эндпоинтам, причём синхронно
    внутри async-обработчика — то есть блокировало весь event loop.
    """
    return WordListManager()


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
