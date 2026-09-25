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
