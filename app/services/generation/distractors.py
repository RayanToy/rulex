"""Этап 3: неверные варианты ответа (дистракторы).

Дистрактор должен быть той же частью речи, что и загаданное слово, не
однокоренным с ним и не повторять другие варианты. Схема ответа
гарантирует только форму (список строк), поэтому лингвистические
проверки одни и те же для структурного ответа и для разбора текста.
"""
import re
from collections.abc import Callable

from app.services.generation.model_calls import ModelCaller, clean_token
from app.services.generation.morphology import get_lemma, get_pos, pos_matches, pos_name

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


def distractors_prompt(word: str) -> str:
    pos = get_pos(word)
    pos_rus = pos_name(pos)
    example = ("дом, лес, река, гора, поле, берег, холм, овраг, поляна, роща"
               if pos == 'NOUN'
               else "бежать, идти, прыгать, ползти, лететь, плыть, ехать, спешить, брести, мчаться")

    return f"""Для теста на словарный запас нужны слова-дистракторы к слову "{word}" ({pos_rus}).

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


class DistractorPicker:
    def __init__(self, llm: ModelCaller, trace: Callable[[str, dict], None] | None = None):
        self.llm = llm
        self._trace = trace or (lambda step, data: None)

    def pick(self, word: str) -> list[str]:
        """До трёх дистракторов к слову."""
        data, text = self.llm.structured(distractors_prompt(word), TOOL_DISTRACTORS, max_tokens=1024)
        self._trace("distractors_response", {"word": word, "structured": data is not None,
                                             "response": data if data is not None else text})

        if data is not None:
            return select_distractors(data.get("distractors", []), word)
        return parse_distractors(text, word)


def extract_list_line(response: str) -> str:
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


def parse_distractors(response: str, word: str) -> list[str]:
    """Разбор текстового ответа в список дистракторов.

    Отличия от прежнего разбора: снимается нумерация, отсеиваются
    повторы (модели охотно выдают одно слово по три раза — у qwen3
    это давало 21% заданий с двумя одинаковыми вариантами) и часть
    речи проверяется по ВСЕМ разборам pymorphy3, а не только по
    самому вероятному: «род» как существительное иначе теряется,
    потому что первым разбором идёт глагольная форма.
    """
    text = extract_list_line(response)
    return select_distractors(re.split(r"[,;\n]+", text), word)


def select_distractors(candidates: list[str], word: str) -> list[str]:
    """Отбор дистракторов из кандидатов — общий для структурного ответа
    и для разбора текста.

    Схема гарантирует форму ответа (список строк), но не лингвистику:
    повторы, другая часть речи и однокоренные с целевым словом
    приходят и в валидном JSON, поэтому проверки те же.
    """
    target_pos = get_pos(word)
    seen_lemmas = {get_lemma(word)}
    distractors = []
    for candidate in candidates:
        candidate = clean_token(str(candidate))
        if not candidate or len(candidate) < 2 or not candidate.isalpha():
            continue
        if candidate == word.lower():
            continue
        if not pos_matches(candidate, target_pos):
            continue

        lemma = get_lemma(candidate)
        if lemma in seen_lemmas:
            continue  # и дубль, и однокоренное с целевым словом
        seen_lemmas.add(lemma)

        distractors.append(candidate)
        if len(distractors) >= 3:
            break

    return distractors
