"""Структурированные ответы модели: строгий tool use и разбор-запаска.

Клиент подменяется фальшивым, поэтому тесты не ходят в сеть и не тратят
деньги. Проверяется контракт: структурный ответ используется напрямую,
текстовый — разбирается запаской, и оба пути дают одинаковый результат.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generator  # noqa: E402
import llm_backends  # noqa: E402


def tool_use(name, payload):
    return SimpleNamespace(type="tool_use", id="t1", name=name, input=payload)


def text(value):
    return SimpleNamespace(type="text", text=value)


class FakeClient:
    """Отвечает заготовленными блоками и запоминает параметры запросов."""

    def __init__(self, *responses, error=None):
        self._responses = list(responses)
        self._error = error
        self.calls = []
        self.messages = self

    def create(self, **params):
        self.calls.append(params)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(content=self._responses.pop(0), stop_reason="end_turn")


def make_generator(client):
    gen = generator.QuestionGenerator.__new__(generator.QuestionGenerator)
    gen.client = client
    gen.model = "test-model"
    gen.word_manager = generator.get_word_manager()
    gen.generation_log = []
    gen.filter_failures = []
    gen.structured_stats = {"tool": 0, "fallback": 0}
    return gen


class TestStructuredCall:
    def test_request_uses_strict_tool_with_auto_choice(self):
        client = FakeClient([tool_use("report_real_words", {"real_words": []})])
        gen = make_generator(client)
        gen._call_llm_structured("промпт", gen.TOOL_REAL_WORDS)

        params = client.calls[0]
        assert params["tools"][0]["strict"] is True
        # Принудительный выбор несовместим с рассуждением у Sonnet 5 / Opus 5
        assert params["tool_choice"] == {"type": "auto"}
        assert "report_real_words" in params["messages"][0]["content"]

    def test_tool_use_answer_is_returned_as_data(self):
        client = FakeClient([tool_use("report_real_words", {"real_words": ["стол"]})])
        gen = make_generator(client)
        data, fallback = gen._call_llm_structured("промпт", gen.TOOL_REAL_WORDS)
        assert data == {"real_words": ["стол"]}
        assert fallback == ""
        assert gen.structured_stats == {"tool": 1, "fallback": 0}

    def test_text_answer_goes_to_fallback_and_is_counted(self):
        client = FakeClient([text("стол, книга")])
        gen = make_generator(client)
        data, fallback = gen._call_llm_structured("промпт", gen.TOOL_REAL_WORDS)
        assert data is None
        assert fallback == "стол, книга"
        assert gen.structured_stats == {"tool": 0, "fallback": 1}

    def test_thinking_block_before_tool_use_is_skipped(self):
        thinking = SimpleNamespace(type="thinking", thinking="")
        client = FakeClient([thinking, tool_use("report_real_words", {"real_words": ["дом"]})])
        gen = make_generator(client)
        data, _ = gen._call_llm_structured("промпт", gen.TOOL_REAL_WORDS)
        assert data == {"real_words": ["дом"]}


class TestRealWordsFilter:
    def test_structured_path(self):
        client = FakeClient([tool_use("report_real_words", {"real_words": ["стол", "книга"]})])
        gen = make_generator(client)
        assert gen._filter_real_words_batch(["жоут", "стол", "книга"]) == ["стол", "книга"]

    def test_markdown_bold_in_text_answer_is_understood(self):
        """Регрессия: шлюз, выбросивший структуру, отвечал `- **стол**`.
        Прежний разбор не снимал звёздочки, слово не совпадало с исходным,
        и реальное слово тихо считалось отвергнутым."""
        client = FakeClient([text("Реальные слова:\n\n- **стол**\n- **книга**\n\nОстальные — мусор.")])
        gen = make_generator(client)
        assert gen._filter_real_words_batch(["жоут", "стол", "книга"]) == ["стол", "книга"]


class TestSuitability:
    def test_structured_verdicts(self):
        gen = make_generator(FakeClient(
            [tool_use("report_suitability", {"suitable": True, "reason": "общеупотребительное"})],
            [tool_use("report_suitability", {"suitable": False, "reason": "топоним"})],
        ))
        assert gen._check_word_suitability("стол") == (True, "общеупотребительное")
        assert gen._check_word_suitability("москва") == (False, "топоним")

    def test_api_failure_fails_closed(self):
        """Раньше сбой API объявлял слово пригодным и пропускал в тест
        топонимы и узкие термины. Теперь сбой = не пригодно."""
        gen = make_generator(FakeClient(error=llm_backends.OllamaError("нет связи")))
        suitable, reason = gen._check_word_suitability("стол")
        assert suitable is False
        assert "Не удалось проверить" in reason

    @pytest.mark.parametrize("answer,expected", [
        ("Подходит: да. Общеупотребительное слово.", True),
        ("ПОДХОДИТ: да", True),
        ("Подходит — да", True),
        ("Не подходит: это топоним.", False),
        ("Подходит: нет", False),
    ])
    def test_text_fallback(self, answer, expected):
        gen = make_generator(FakeClient([text(answer)]))
        assert gen._check_word_suitability("слово")[0] is expected


class TestDistractors:
    def test_structured_path_still_validates_linguistics(self):
        """Схема гарантирует форму, но не лингвистику: повторы и чужая
        часть речи приходят и в валидном JSON."""
        payload = {"distractors": ["племя", "племя", "бежать", "народ", "житель", "предок"]}
        gen = make_generator(FakeClient([tool_use("report_distractors", payload)]))
        result = gen._get_distractors("кроманьонец", 6)
        assert result == ["племя", "народ", "житель"]

    def test_text_fallback(self):
        gen = make_generator(FakeClient([text("человек, житель, племя, народ")]))
        assert gen._get_distractors("кроманьонец", 6) == ["человек", "житель", "племя"]


class TestOllamaAdapter:
    """Адаптер переводит строгий инструмент в параметр format Ollama
    и возвращает блок tool_use — вызывающему коду всё равно, облако или нет."""

    def _client(self, monkeypatch, reply_content):
        sent = {}

        def fake_post(self, path, payload):
            sent["path"], sent["payload"] = path, payload
            return {"message": {"content": reply_content},
                    "prompt_eval_count": 10, "eval_count": 5, "done_reason": "stop"}

        monkeypatch.setattr(llm_backends.OllamaClient, "post", fake_post)
        return llm_backends.OllamaClient(model="local"), sent

    def test_tool_becomes_format_and_tool_use(self, monkeypatch):
        client, sent = self._client(monkeypatch, json.dumps({"real_words": ["стол"]}))
        tool = generator.QuestionGenerator.TOOL_REAL_WORDS
        response = client.messages.create(model="local", max_tokens=100,
                                          messages=[{"role": "user", "content": "x"}],
                                          tools=[tool], tool_choice={"type": "auto"})
        assert sent["payload"]["format"] == tool["input_schema"]
        block = response.content[0]
        assert block.type == "tool_use"
        assert block.name == "report_real_words"
        assert block.input == {"real_words": ["стол"]}

    def test_invalid_json_degrades_to_text(self, monkeypatch):
        client, _ = self._client(monkeypatch, "стол, книга")
        response = client.messages.create(
            model="local", max_tokens=100, messages=[{"role": "user", "content": "x"}],
            tools=[generator.QuestionGenerator.TOOL_REAL_WORDS])
        assert response.content[0].type == "text"
        assert response.content[0].text == "стол, книга"

    def test_without_tools_no_format_is_sent(self, monkeypatch):
        client, sent = self._client(monkeypatch, "обычный ответ")
        client.messages.create(model="local", max_tokens=100,
                               messages=[{"role": "user", "content": "x"}])
        assert "format" not in sent["payload"]


class TestEvalCache:
    """Кэш харнесса должен воспроизводить ответ в той же форме.

    Раньше кэшировался только текст: на повторном прогоне структурный
    ответ возвращался текстом, и харнесс показал бы 0% структурных
    ответов — то есть измерял бы сам себя, а не модель.
    """

    @pytest.fixture
    def recorder_cls(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(ROOT / "eval"))
        import run_eval
        monkeypatch.setattr(run_eval, "CACHE_PATH", tmp_path / "cache.json")
        return run_eval.LLMRecorder

    def test_tool_use_survives_cache_roundtrip(self, recorder_cls):
        client = FakeClient([tool_use("report_real_words", {"real_words": ["стол"]})])
        recorder = recorder_cls(client, use_cache=True)
        request = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}],
                   "tools": [generator.QuestionGenerator.TOOL_REAL_WORDS]}

        live = recorder(**request)
        cached = recorder(**request)          # второй раз — из кэша, сеть не трогаем

        assert live.content[0].type == "tool_use"
        assert cached.content[0].type == "tool_use"
        assert cached.content[0].input == {"real_words": ["стол"]}
        assert len(client.calls) == 1

    def test_structured_and_plain_requests_do_not_collide(self, recorder_cls):
        client = FakeClient([tool_use("report_real_words", {"real_words": ["стол"]})],
                            [text("стол")])
        recorder = recorder_cls(client, use_cache=True)
        base = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}]}

        recorder(**base, tools=[generator.QuestionGenerator.TOOL_REAL_WORDS])
        plain = recorder(**base)

        assert plain.content[0].type == "text"
        assert len(client.calls) == 2
