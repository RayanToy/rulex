"""Исполнение пачки запросов и хранилище вердиктов.

Batches API проверяется на фальшивом клиенте: доступный шлюз этот эндпоинт
не поддерживает, а прямой ключ Anthropic без средств. Проверяется контракт,
на котором легко ошибиться: результаты приходят в произвольном порядке,
неудачный запрос не должен превращаться в вердикт, а отсутствие эндпоинта —
в падение задачи.
"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from anthropic import NotFoundError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services import batches as batch_runner  # noqa: E402
from app.services import wordlists  # noqa: E402
from app.services.generation.realness import RealnessFilter  # noqa: E402
from app.services.llm import OllamaError  # noqa: E402


def text_content(value):
    return [SimpleNamespace(type="text", text=value)]


class FakeBatches:
    """Имитация client.messages.batches."""

    def __init__(self, outcomes, polls_until_done=2, shuffle=True, create_error=None):
        self.outcomes = outcomes          # custom_id -> ("succeeded", content) | ("errored", None) ...
        self.polls_left = polls_until_done
        self.shuffle = shuffle
        self.create_error = create_error
        self.cancelled = []
        self.submitted = None

    def create(self, requests):
        if self.create_error is not None:
            raise self.create_error
        self.submitted = requests
        return SimpleNamespace(id="batch-1", processing_status="in_progress")

    def retrieve(self, batch_id):
        self.polls_left -= 1
        status = "ended" if self.polls_left <= 0 else "in_progress"
        return SimpleNamespace(id=batch_id, processing_status=status,
                               request_counts=SimpleNamespace(processing=max(self.polls_left, 0)))

    def results(self, batch_id):
        items = []
        for custom_id, (kind, content) in self.outcomes.items():
            result = SimpleNamespace(type=kind)
            if kind == "succeeded":
                result.message = SimpleNamespace(content=content)
            items.append(SimpleNamespace(custom_id=custom_id, result=result))
        if self.shuffle:
            random.Random(7).shuffle(items)   # API не гарантирует порядок
        return items

    def cancel(self, batch_id):
        self.cancelled.append(batch_id)


class FakeClient:
    def __init__(self, batches=None, create=None):
        self.messages = SimpleNamespace(create=create or self._no_create)
        if batches is not None:
            self.messages.batches = batches

    @staticmethod
    def _no_create(**_):
        raise AssertionError("синхронный путь не должен вызываться")


def not_found():
    request = httpx.Request("POST", "https://gateway/v1/messages/batches")
    response = httpx.Response(404, request=request)
    return NotFoundError("Invalid URL (POST /v1/messages/batches)", response=response, body=None)


REQUESTS = {f"chunk-{i}": {"model": "m", "max_tokens": 10,
                           "messages": [{"role": "user", "content": str(i)}]} for i in range(5)}


class TestBatches:
    def test_results_are_matched_by_custom_id_not_position(self):
        outcomes = {cid: ("succeeded", text_content(f"ответ на {cid}")) for cid in REQUESTS}
        client = FakeClient(FakeBatches(outcomes))
        results = batch_runner.run_via_batches(client, REQUESTS, sleep=lambda _s: None)
        for cid in REQUESTS:
            assert results[cid][0].text == f"ответ на {cid}"

    def test_failed_requests_become_none(self):
        outcomes = {
            "chunk-0": ("succeeded", text_content("ок")),
            "chunk-1": ("errored", None),
            "chunk-2": ("expired", None),
            "chunk-3": ("canceled", None),
            # chunk-4 не вернулся вовсе
        }
        client = FakeClient(FakeBatches(outcomes))
        results = batch_runner.run_via_batches(client, REQUESTS, sleep=lambda _s: None)
        assert results["chunk-0"] is not None
        assert all(results[c] is None for c in ("chunk-1", "chunk-2", "chunk-3", "chunk-4"))

    def test_polls_until_ended(self):
        outcomes = {cid: ("succeeded", text_content("x")) for cid in REQUESTS}
        batches = FakeBatches(outcomes, polls_until_done=4)
        naps = []
        batch_runner.run_via_batches(FakeClient(batches), REQUESTS, poll_seconds=30, sleep=naps.append)
        assert naps == [30, 30, 30, 30]

    def test_timeout_cancels_batch(self, monkeypatch):
        batches = FakeBatches({}, polls_until_done=10**6)
        clock = iter(range(0, 10**7, 100))
        monkeypatch.setattr(batch_runner.time, "monotonic", lambda: next(clock))
        with pytest.raises(TimeoutError):
            batch_runner.run_via_batches(FakeClient(batches), REQUESTS,
                                         timeout_seconds=250, sleep=lambda _s: None)
        assert batches.cancelled == ["batch-1"]

    def test_request_params_are_passed_through(self):
        outcomes = {cid: ("succeeded", text_content("x")) for cid in REQUESTS}
        batches = FakeBatches(outcomes)
        batch_runner.run_via_batches(FakeClient(batches), REQUESTS, sleep=lambda _s: None)
        submitted = {r["custom_id"]: r["params"] for r in batches.submitted}
        assert submitted["chunk-3"]["messages"][0]["content"] == "3"


class TestFallback:
    def test_gateway_404_falls_back_to_sync(self):
        """Так ведёт себя router.cheap: POST /v1/messages/batches -> 404."""
        calls = []
        client = FakeClient(FakeBatches({}, create_error=not_found()),
                            create=lambda **p: calls.append(p) or SimpleNamespace(content=text_content("ок")))
        results, mode = batch_runner.run_requests(client, REQUESTS)
        assert mode == "sync"
        assert len(calls) == len(REQUESTS)
        assert all(results[c] is not None for c in REQUESTS)

    def test_client_without_batches_falls_back_to_sync(self):
        """У адаптера Ollama эндпоинта батчей нет вовсе."""
        client = FakeClient(create=lambda **p: SimpleNamespace(content=text_content("ок")))
        _, mode = batch_runner.run_requests(client, REQUESTS)
        assert mode == "sync"

    def test_sync_failure_of_one_request_is_isolated(self):
        def create(**params):
            if params["messages"][0]["content"] == "2":
                raise OllamaError("сбой")
            return SimpleNamespace(content=text_content("ок"))

        results = batch_runner.run_sync(FakeClient(create=create), REQUESTS)
        assert results["chunk-2"] is None
        assert sum(r is not None for r in results.values()) == len(REQUESTS) - 1


class TestVerdictStore:
    @pytest.fixture(autouse=True)
    def isolated_store(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wordlists, "VERDICTS_PATH", tmp_path / "verdicts.tsv")
        wordlists.get_verdicts.cache_clear()
        yield
        wordlists.get_verdicts.cache_clear()

    def test_roundtrip(self):
        wordlists.append_verdicts([("стол", "real", "dict"), ("жоут", "artifact", "llm:m:sync")])
        assert wordlists.get_verdicts() == {"стол": "real", "жоут": "artifact"}

    def test_append_is_visible_without_restart(self):
        wordlists.append_verdicts([("стол", "real", "dict")])
        assert "стол" in wordlists.get_verdicts()
        wordlists.append_verdicts([("книга", "real", "dict")])
        assert "книга" in wordlists.get_verdicts()

    def test_cascade_uses_stored_verdicts_before_model(self):
        wordlists.append_verdicts([
            ("суэссоя", "artifact", "llm:m:sync"),      # мусор, модель уже решила
            ("робототехник", "real", "llm:m:sync"),     # вне словарей, модель уже решила
        ])
        realness = RealnessFilter(llm=None, word_manager=wordlists.get_word_manager())

        def model_must_not_be_called(*_a, **_k):
            raise AssertionError("вердикт сохранён — модель вызывать не нужно")
        realness.filter_batch = model_must_not_be_called

        result = realness.confirm(["стол", "суэссоя", "робототехник"])
        assert sorted(result) == ["робототехник", "стол"]
