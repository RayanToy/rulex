"""Этап 5: однозначно ли задание — его решает модель, не зная ответа.

Модель получает толкование и варианты в алфавитном порядке и называет
все, что подходят. Проверка ловит то, чего не видят остальные этапы:
дистракторы, которые подходят под толкование почти так же хорошо, как
ответ. Промпт дистракторов запрещает синонимы, но модель их всё равно
даёт: к «вписаться» с толкованием «разместиться в ограниченном
пространстве» — «поместиться» и «уместиться». Ученик, выбравший их,
получил бы «неверно» при верном по смыслу ответе.

Решение:
- решатель не назвал загаданное слово — толкование на него не указывает,
  задание отбраковывается;
- назвал дистракторы — они убираются; если осталось меньше двух,
  задание отбраковывается.

Сбой модели — тоже отбраковка: непроверенное задание в банк не попадает.
"""
import re
from collections.abc import Callable

from app.services.generation.model_calls import ModelCaller, clean_token

TOOL_MATCHING_OPTIONS = {
    "name": "report_matching_options",
    "description": "Сообщить, какие варианты ответа подходят под толкование.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "matching": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Все подходящие варианты, в написании из списка",
            },
            "reason": {"type": "string", "description": "Краткое объяснение"},
        },
        "required": ["matching", "reason"],
        "additionalProperties": False,
    },
}

MIN_DISTRACTORS = 2


def answerability_prompt(definition: str, options: list[str]) -> str:
    return f"""Проверь тестовое задание на словарный запас: по толкованию нужно выбрать слово.

Толкование: «{definition}»

Варианты ответа: {", ".join(options)}

Назови ВСЕ варианты, которые подходят под это толкование: и точный ответ, и те, что подходят почти так же хорошо — синонимы и близкие по смыслу слова. Если толкование однозначно указывает на одно слово, назови только его."""


class AnswerabilityCheck:
    def __init__(self, llm: ModelCaller, trace: Callable[[str, dict], None] | None = None):
        self.llm = llm
        self._trace = trace or (lambda step, data: None)

    def filter_distractors(self, word: str, definition: str, distractors: list[str]) -> list[str]:
        """Дистракторы, которые не подходят под толкование. ValueError — задание неоднозначно."""
        options = sorted([word, *distractors])
        data, text = self.llm.structured(answerability_prompt(definition, options),
                                         TOOL_MATCHING_OPTIONS, max_tokens=512, effort="low")
        if data is not None:
            matching = {clean_token(str(o)) for o in data.get("matching", [])}
            reason = str(data.get("reason") or "")
        else:
            # Запаска: какие варианты упомянуты в текстовом ответе
            mentioned = set(re.findall(r"[а-яё]+", text.lower()))
            matching = {o for o in options if o in mentioned}
            reason = text[:200]

        ambiguous = [d for d in distractors if d.lower() in matching]
        kept = [d for d in distractors if d.lower() not in matching]
        self._trace("answerability", {"matching": sorted(matching), "dropped": ambiguous,
                                      "reason": reason})

        if word.lower() not in matching:
            raise ValueError(f"Толкование не указывает на '{word}': решатель выбрал {sorted(matching)}")
        if len(kept) < MIN_DISTRACTORS:
            raise ValueError(f"Задание к '{word}' неоднозначно: подходят и {ambiguous}")
        return kept
