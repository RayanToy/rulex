"""Этап 4: толкование — сам текст задания.

Главный риск — толкование, по которому ответ очевиден: в нём стоит само
слово, однокоренное с ним или один из вариантов ответа. Такое
толкование отправляется модели на правку, а если правки не помогли —
не сохраняется вовсе.
"""
import re
from collections.abc import Callable

from app.services.generation.model_calls import ModelCaller
from app.services.generation.morphology import get_lemma, get_pos, pos_name

# Признаки того, что модель не дала толкование, а отказалась или
# стала рассуждать о запросе. Проверяется только начало ответа.
REFUSAL_MARKERS = (
    "я не могу", "не могу выполнить", "не могу создать", "к сожалению",
    "не подходит для теста", "это узкоспециальн", "как языковая модель",
)

CORRECTION_ATTEMPTS = 3


def definition_prompt(word: str, forbidden: str) -> str:
    return f"""Напиши толкование для слова "{word}" ({pos_name(get_pos(word))}) для теста на словарный запас школьника.

ВАЖНО:
1. Толкование должно быть кратким и понятным
2. НЕ используй слова: {forbidden} и однокоренные им
3. НЕ используй специальные термины
4. НЕ ссылайся на географические названия или национальности
5. Используй только простые, общеупотребительные слова
6. Толкование должно однозначно указывать на слово "{word}"

Напиши только само толкование, без целевого слова и тире:"""


def correction_prompt(definition: str, forbidden: str) -> str:
    return f"""Исправь толкование так, чтобы не использовать слова, однокоренные с: {forbidden}

Текущее толкование: "{definition}"

Сохрани смысл, но перефразируй. Напиши только исправленное толкование:"""


def shares_root(definition: str, words: list[str]) -> bool:
    """Есть ли в толковании слово, однокоренное с одним из запрещённых.

    Корень определяется грубо — совпадением леммы или её первых четырёх
    букв. Это ловит «лесной» при слове «лес», но даёт ложные срабатывания
    («столица» при «стол»). Здесь это приемлемо: срабатывание означает лишь
    лишнюю правку, а окончательное решение принимает validate_definition.
    """
    forbidden = [get_lemma(w) for w in words]
    for token in re.findall(r'[а-яёА-ЯЁ]+', definition.lower()):
        lemma = get_lemma(token)
        for other in forbidden:
            if lemma == other:
                return True
            if len(lemma) > 3 and len(other) > 3 and lemma[:4] == other[:4]:
                return True
    return False


def validate_definition(definition: str, word: str) -> None:
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
    if any(marker in head for marker in REFUSAL_MARKERS):
        raise ValueError(f"Модель отказалась толковать '{word}'")

    target = get_lemma(word)
    for token in re.findall(r"[а-яёА-ЯЁ]+", definition):
        if get_lemma(token.lower()) == target:
            raise ValueError(f"Толкование для '{word}' содержит само слово")


def _clean(definition: str) -> str:
    return definition.strip().lstrip('-—').strip()


class DefinitionWriter:
    def __init__(self, llm: ModelCaller, trace: Callable[[str, dict], None] | None = None):
        self.llm = llm
        self._trace = trace or (lambda step, data: None)

    def write(self, word: str, distractors: list[str]) -> str:
        """Толкование, в котором нет ни слова, ни вариантов ответа, ни однокоренных."""
        forbidden_words = [word] + distractors
        forbidden = ', '.join(forbidden_words)

        definition = _clean(self.llm.text(definition_prompt(word, forbidden), max_tokens=150))
        self._trace("definition_initial", {"word": word, "definition": definition})

        for attempt in range(CORRECTION_ATTEMPTS):
            if not shares_root(definition, forbidden_words):
                break
            definition = _clean(self.llm.text(correction_prompt(definition, forbidden), max_tokens=150))
            self._trace("definition_corrected", {"attempt": attempt + 1, "definition": definition})

        validate_definition(definition, word)
        return definition
