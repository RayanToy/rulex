"""Исполнение пачки запросов к модели: через Batches API или по одному.

Batches API (`POST /v1/messages/batches`) стоит вдвое дешевле обычных
вызовов и не упирается в лимиты на одновременные запросы — а замер показал,
что параллельные запросы через шлюз работают впятеро медленнее
последовательных. Цена — задержка: батч выполняется от минут до суток.
Поэтому исполнитель предназначен для офлайн-задач, а не для обработчика
запроса.

Batches API есть не везде. Шлюз router.cheap отвечает на него
`404 Invalid URL (POST /v1/messages/batches)`, у локальной Ollama такого
эндпоинта нет вовсе. В этих случаях исполнитель сам переходит на обычные
вызовы по одному, и вызывающий код об этом узнаёт из возвращаемого режима.

ВНИМАНИЕ: ветка Batches API покрыта тестами на фальшивом клиенте, но против
настоящего эндпоинта не прогонялась: доступный шлюз его не поддерживает,
а у прямого ключа Anthropic нет средств.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from anthropic import APIConnectionError, APIError, APIStatusError, NotFoundError, PermissionDeniedError

from llm_backends import OllamaError

# Ответ на один запрос: блоки content либо None, если запрос не выполнен.
Results = dict[str, list | None]


class BatchesUnavailable(RuntimeError):
    """Batches API недоступен на этом клиенте — нужен обычный путь."""


def run_via_batches(
    client,
    requests: dict[str, dict],
    poll_seconds: float = 30.0,
    timeout_seconds: float = 24 * 3600,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] = lambda _msg: None,
) -> Results:
    """Отправить запросы одним батчем и дождаться результатов.

    requests: custom_id -> параметры messages.create.
    Результаты приходят в ПРОИЗВОЛЬНОМ порядке, поэтому сопоставляются
    строго по custom_id, а не по позиции.
    """
    batches = getattr(getattr(client, "messages", None), "batches", None)
    if batches is None:
        raise BatchesUnavailable("у клиента нет messages.batches")

    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    try:
        batch = batches.create(requests=[
            Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**params))
            for custom_id, params in requests.items()
        ])
    except (NotFoundError, PermissionDeniedError) as exc:
        raise BatchesUnavailable(f"{type(exc).__name__}: {exc}") from exc

    progress(f"батч {batch.id} создан, запросов: {len(requests)}")
    started = time.monotonic()
    while batch.processing_status != "ended":
        if time.monotonic() - started > timeout_seconds:
            batches.cancel(batch.id)
            raise TimeoutError(f"батч {batch.id} не завершился за {timeout_seconds} с")
        sleep(poll_seconds)
        batch = batches.retrieve(batch.id)
        counts = getattr(batch, "request_counts", None)
        if counts is not None:
            progress(f"батч {batch.id}: {batch.processing_status}, "
                     f"в работе {getattr(counts, 'processing', '?')}")

    results: Results = {custom_id: None for custom_id in requests}
    for item in batches.results(batch.id):
        outcome = item.result
        # errored / canceled / expired оставляем None: вызывающий код
        # не должен записывать вердикт по запросу, который не выполнен.
        if getattr(outcome, "type", None) == "succeeded":
            results[item.custom_id] = list(outcome.message.content)
    return results


def run_sync(
    client,
    requests: dict[str, dict],
    progress: Callable[[str], None] = lambda _msg: None,
) -> Results:
    """Те же запросы по одному. Сбой отдельного запроса — None, а не падение."""
    results: Results = {}
    for n, (custom_id, params) in enumerate(requests.items(), 1):
        try:
            results[custom_id] = list(client.messages.create(**params).content)
        except (APIStatusError, APIConnectionError, APIError, OllamaError) as exc:
            progress(f"{custom_id}: {type(exc).__name__}")
            results[custom_id] = None
        if n % 10 == 0 or n == len(requests):
            progress(f"выполнено {n}/{len(requests)}")
    return results


def run_requests(
    client,
    requests: dict[str, dict],
    prefer_batches: bool = True,
    progress: Callable[[str], None] = lambda _msg: None,
    **batch_kwargs,
) -> tuple[Results, str]:
    """Выполнить запросы и сообщить, каким путём: "batches" или "sync"."""
    if prefer_batches:
        try:
            return run_via_batches(client, requests, progress=progress, **batch_kwargs), "batches"
        except BatchesUnavailable as exc:
            progress(f"Batches API недоступен ({exc}); перехожу на вызовы по одному")
    return run_sync(client, requests, progress=progress), "sync"
