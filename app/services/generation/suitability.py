"""Этап 2: годится ли реальное слово для теста на словарный запас.

Отсекает то, что знает не каждый школьник: топонимы, этнонимы, узкие
термины. Сбой модели означает «не годится»: лучше не пустить слово
в тест, чем пустить непроверенное.
"""
import re
from collections.abc import Callable

from anthropic import APIConnectionError, APIError, APIStatusError

from app.services.generation.model_calls import ModelCaller
from app.services.llm import OllamaError

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


def suitability_prompt(word: str) -> str:
    return f"""Оцени, подходит ли слово "{word}" для теста на СЛОВАРНЫЙ ЗАПАС школьника.

Слово НЕ подходит, если это:
- Название места (город, страна, регион, река, гора)
- Название народа или национальности (латыш, немец, татарин)
- Прилагательное от географического названия (московский, тверской, балтийский)
- Узкоспециальный термин (медицинский, юридический, технический)
- Устаревшее или диалектное слово
- Имя собственное
- Слово, требующее специальных знаний для понимания

Слово ПОДХОДИТ, если это общеупотребительное слово, значение которого можно объяснить без специальных знаний."""


class SuitabilityCheck:
    def __init__(self, llm: ModelCaller, trace: Callable[[str, dict], None] | None = None):
        self.llm = llm
        self._trace = trace or (lambda step, data: None)

    def check(self, word: str) -> tuple[bool, str]:
        """Проверка слова через LLM на пригодность для теста словарного запаса"""
        try:
            # Прежний разбор: `"ПОДХОДИТ: да" in response.lower()` — левая часть
            # никогда не срабатывала (верхний регистр ищется в нижнем), а на
            # ответе «Подходит — да» не срабатывала и правая.
            data, text = self.llm.structured(
                suitability_prompt(word), TOOL_SUITABILITY, max_tokens=512, effort="low")
        except (APIStatusError, APIConnectionError, APIError, OllamaError) as e:
            # Раньше здесь возвращалось (True, ...): сбой API объявлял слово
            # пригодным и молча пропускал в тест топонимы и узкие термины.
            # Та же ошибка, что была в батч-фильтре; закрываемся так же.
            self._trace("word_check_error", {"word": word, "error": str(e)})
            return False, f"Не удалось проверить: {type(e).__name__}"

        if data is not None:
            is_suitable = bool(data.get("suitable"))
            reason = str(data.get("reason") or ("OK" if is_suitable else "Не подходит"))
        else:
            # Запаска: ищем явное «да»/«нет» в тексте
            lowered = text.lower()
            is_suitable = bool(re.search(r"подходит\W{0,5}да\b", lowered)) and "не подходит" not in lowered
            reason = text[:200] or "Не удалось разобрать ответ"

        self._trace("word_check", {"word": word, "suitable": is_suitable, "reason": reason})
        return is_suitable, reason
