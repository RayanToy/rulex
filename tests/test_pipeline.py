"""Сборка генератора из этапов.

Снимок (test_generation_snapshot.py) проверяет поведение генератора,
собранного с фальшивой моделью. Здесь то, чего снимок не видит: сборка
по умолчанию из общих фабрик и изоляция параллельных задач.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.generation import model_calls, pipeline  # noqa: E402


def test_default_wiring_uses_shared_factories(monkeypatch):
    client, words = object(), SimpleNamespace(relative_lists={})
    monkeypatch.setattr(model_calls, "get_client", lambda: client)
    monkeypatch.setattr(pipeline, "get_word_manager", lambda: words)
    monkeypatch.setenv("RULEX_MODEL_GENERATION", "env-model")

    gen = pipeline.QuestionGenerator()

    assert gen.client is client
    assert gen.model == "env-model"
    assert gen.word_manager is words
    # Все этапы ходят в модель через один ModelCaller: общий системный
    # промпт и общий счётчик структурных ответов
    assert all(stage.llm is gen.llm
               for stage in (gen.realness, gen.suitability, gen.distractors, gen.definitions))


def test_worker_shares_client_but_not_state():
    """Параллельная задача получает свой журнал и счётчики, иначе задачи
    затирали бы их друг у друга; клиент, модель и словари — общие."""
    gen = pipeline.QuestionGenerator(llm=model_calls.ModelCaller(client=object(), model="m"),
                                     word_manager=SimpleNamespace(relative_lists={}))
    worker = gen._worker()

    assert worker.client is gen.client
    assert worker.model == gen.model
    assert worker.word_manager is gen.word_manager
    assert worker.structured_stats is not gen.structured_stats

    worker.generation_log.append({"step": "start"})
    assert gen.generation_log == []


def test_stage_trace_lands_in_generator_log():
    gen = pipeline.QuestionGenerator(llm=model_calls.ModelCaller(client=object(), model="m"),
                                     word_manager=SimpleNamespace(relative_lists={}))
    gen.suitability._trace("word_check", {"word": "стол"})
    assert gen.generation_log == [{"step": "word_check", "word": "стол"}]


class TestFrequencyBands:
    """Частотность — по словарю, а не по позиции в перемешанном списке."""

    @staticmethod
    def bands(freqs):
        from app.services.generation.frequency import frequency_bands
        return frequency_bands(list(freqs), SimpleNamespace(get_sharov_frequency=freqs.get))

    def test_absent_from_dictionary_is_low(self):
        assert self.bands({"стол": 100.0, "свояченица": 0.0})["свояченица"] == "low"

    def test_known_words_split_by_median(self):
        result = self.bands({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
        assert [result[w] for w in "abcd"] == ["medium", "medium", "high", "high"]

    def test_label_follows_the_word_not_its_position(self):
        freqs = {"свояченица": 0.0, "книга": 250.0, "лампа": 20.0}
        forward = self.bands(freqs)
        backward = self.bands(dict(reversed(list(freqs.items()))))
        assert forward == backward
        assert forward["свояченица"] == "low" and forward["книга"] == "high"

    def test_nothing_known_means_all_low(self):
        assert set(self.bands({"а": 0.0, "б": 0.0}).values()) == {"low"}
