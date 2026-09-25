"""Общий клиент модели для всех этапов генерации.

Этапы ходят в модель только через ModelCaller: у всех один системный
промпт, один способ достать текст из ответа и один способ получить ответ
по JSON-схеме. Отдельный этап не может забыть пропустить блок
рассуждения или посчитать структурный ответ.
"""
import functools
import os
import re

from dotenv import load_dotenv

load_dotenv()

# Контекст для модели о назначении теста — общий для всех этапов
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


@functools.lru_cache(maxsize=1)
def get_client():
    """Один клиент на процесс — иначе теряется пул соединений.

    Какой именно бэкенд, решают переменные окружения (см. app/services/llm.py):
    облачный Anthropic, совместимый шлюз через ANTHROPIC_BASE_URL или
    локальная модель через Ollama. Интерфейс у них одинаковый, поэтому
    остальной код о разнице не знает.
    """
    from app.services.llm import build_client

    return build_client()


def clean_token(token: str) -> str:
    """Слово из текстового ответа без markdown и пунктуации.

    Модели оформляют списки жирным (`**стол**`) и кодом (`` `стол` ``).
    Прежний разбор снимал только точки и запятые, и выделенное слово
    не совпадало с исходным — реальное слово тихо считалось отвергнутым.
    """
    token = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", token)
    return token.strip().strip("*_`'\"«».,;:()[]").strip().lower()


class ModelCaller:
    """Модель, системный промпт и счётчик структурных ответов."""

    def __init__(self, client=None, model: str | None = None):
        # Клиент общий на процесс (get_client), а счётчики — свои у каждого
        # экземпляра: параллельные задачи не делят изменяемое состояние.
        self.client = client if client is not None else get_client()
        # Модель вынесена в окружение: прежний claude-sonnet-4-20250514
        # снят с обслуживания и отвечал 404 на каждый вызов.
        self.model = model or os.getenv("RULEX_MODEL_GENERATION", "claude-sonnet-5")
        # Сколько раз ответ пришёл структурой, а сколько раз пришлось
        # откатиться на разбор текста. Совместимые шлюзы умеют молча
        # выбрасывать параметры запроса, и без счётчика этого не видно.
        self.structured_stats = {"tool": 0, "fallback": 0}

    def text(self, prompt: str, max_tokens: int = 500, effort: str | None = None) -> str:
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
            "system": SYSTEM_CONTEXT,
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

    def structured(
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
        params = self.structured_params(prompt, tool, max_tokens, effort)
        response = self.client.messages.create(**params)
        return self.parse_structured(response.content, tool)

    def structured_params(
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
            "system": SYSTEM_CONTEXT,
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

    def parse_structured(self, content, tool: dict) -> tuple[dict | None, str]:
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
