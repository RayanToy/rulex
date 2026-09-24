"""Бэкенды генерации: облачный Anthropic и локальный через Ollama.

Адаптер намеренно повторяет интерфейс Anthropic SDK — метод
`messages.create(...)` и ответ с полями `content`, `usage`, `stop_reason`.
Благодаря этому ни пайплайн генерации, ни eval-харнесс не знают, какой
бэкенд под ними: подменяется ровно объект клиента.

Подключение к соседней машине: Ollama по умолчанию слушает только
127.0.0.1, поэтому на ней нужно выставить OLLAMA_HOST=0.0.0.0:11434
и открыть TCP-порт 11434 в брандмауэре, иначе адрес не поможет.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from types import SimpleNamespace

# Рассуждающие модели пишут ход мыслей перед ответом. Он идёт по-английски
# даже на русский запрос, съедает лимит токенов и обрезает настоящий ответ.
# Встречаются оба случая: парный тег и только закрывающий (открывающий
# ставит шаблон модели, и в ответе его не видно).
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Убирает ход рассуждений, оставляя ответ.

    Если закрывающего тега нет вовсе (модель не закончила рассуждать),
    текст возвращается как есть: пустая строка была бы хуже — она
    неотличима от «модель промолчала», а так дефект виден в ответе.
    """
    without_pairs = _THINK_BLOCK.sub("", text)
    parts = _THINK_CLOSE.split(without_pairs)
    return parts[-1].strip() if len(parts) > 1 else without_pairs.strip()


class OllamaError(RuntimeError):
    pass


class _OllamaMessages:
    """Реализует messages.create в терминах Ollama /api/chat."""

    def __init__(self, backend: OllamaClient):
        self._backend = backend

    def create(self, *, model=None, max_tokens=None, system=None,
               messages=None, output_config=None, tools=None, **_ignored):
        # output_config.effort — облачное понятие, у Ollama его нет.
        # Игнорируем осознанно, а не роняем запрос.
        #
        # Инструменты переводим в собственный механизм Ollama: параметр
        # format принимает JSON-схему и ограничивает вывод ею. Ответ
        # возвращаем блоком tool_use — так вызывающий код одинаков для
        # облака и локальной модели.
        tool = tools[0] if tools else None
        payload = {
            "model": model or self._backend.model,
            "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + list(messages or [])
            ),
            "stream": False,
            "think": self._backend.think,
            "options": {
                "temperature": self._backend.temperature,
                "num_predict": max_tokens or 700,
                "num_ctx": self._backend.num_ctx,
                "seed": self._backend.seed,
            },
        }
        if tool is not None:
            payload["format"] = tool["input_schema"]

        data = self._backend.post("/api/chat", payload)
        text = strip_thinking((data.get("message") or {}).get("content", ""))

        content = [SimpleNamespace(type="text", text=text)]
        if tool is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            # Невалидный JSON отдаём текстом: пусть решает разбор-запаска,
            # а не падает весь запрос.
            if isinstance(parsed, dict):
                content = [SimpleNamespace(type="tool_use", id="ollama",
                                           name=tool["name"], input=parsed)]

        return SimpleNamespace(
            # Форма ответа как у Anthropic SDK: список блоков с типом.
            content=content,
            usage=SimpleNamespace(
                input_tokens=data.get("prompt_eval_count", 0) or 0,
                output_tokens=data.get("eval_count", 0) or 0,
            ),
            stop_reason=data.get("done_reason") or "end_turn",
            model=payload["model"],
        )


class OllamaClient:
    """Клиент Ollama с интерфейсом, совместимым с Anthropic SDK."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "",
                 num_ctx: int = 8192, temperature: float = 0.0,
                 timeout: int = 300, seed: int = 42, think: bool = False):
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.seed = seed
        self.think = think
        self.messages = _OllamaMessages(self)

    def post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Ollama недоступна на {self.host}: {exc}. "
                "Запущен ли сервис? На удалённой машине нужен "
                "OLLAMA_HOST=0.0.0.0:11434 и открытый порт 11434."
            ) from exc

    def health(self) -> dict:
        """Какие модели загружены и есть ли нужная."""
        request = urllib.request.Request(f"{self.host}/api/tags")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                tags = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(f"Ollama недоступна на {self.host}: {exc}") from exc

        names = [m["name"] for m in tags.get("models", [])]
        available = any(
            n == self.model or n.startswith(self.model + ":") for n in names
        )
        return {"models": names, "model_available": available}


def build_client():
    """Клиент по настройкам окружения.

    RULEX_LLM_BACKEND=ollama  -> локальная модель
    иначе                     -> Anthropic (с учётом ANTHROPIC_BASE_URL)
    """
    backend = (os.getenv("RULEX_LLM_BACKEND") or "anthropic").strip().lower()

    if backend == "ollama":
        return OllamaClient(
            host=os.getenv("RULEX_OLLAMA_HOST") or "http://localhost:11434",
            model=os.getenv("RULEX_MODEL_GENERATION") or "",
            num_ctx=int(os.getenv("RULEX_OLLAMA_NUM_CTX") or 8192),
            timeout=int(os.getenv("RULEX_OLLAMA_TIMEOUT") or 300),
        )

    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found")
    return Anthropic(
        api_key=api_key,
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
