"""Наполнение пустого банка готовым набором (демо без ключа модели)."""
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import database, seed  # noqa: E402

BANK = {"questions": [
    {"target_word": "кот", "definition": "Домашнее животное —", "correct_answer": "кот",
     "distractors": ["стул", "окно", "ветер"], "part_of_speech": "NOUN", "word_class": 5,
     "frequency_type": "high", "generation_log": [{"step": "start", "word": "кот"}]},
    {"target_word": "бежать", "definition": "Быстро двигаться —", "correct_answer": "бежать",
     "distractors": ["стоять", "лежать"], "part_of_speech": "INFN", "word_class": 5,
     "frequency_type": "low"},
]}


def seed_into(tmp_path, *bank_files):
    """Засеять временную базу файлами по очереди; число добавленных на каждом шаге."""
    url = f"sqlite:///{tmp_path / 'seed.db'}"
    database.migrate(url)

    async def run():
        engine = create_async_engine(url.replace("sqlite://", "sqlite+aiosqlite://"))
        try:
            factory = async_sessionmaker(engine, expire_on_commit=False)
            return [await seed.seed_question_bank(f, factory) for f in bank_files]
        finally:
            await engine.dispose()

    return url, asyncio.run(run())


def write_bank(path, questions):
    path.write_text(json.dumps({"questions": questions}, ensure_ascii=False), encoding="utf-8")
    return path


def test_empty_bank_is_seeded_once(tmp_path):
    bank_file = write_bank(tmp_path / "bank.json", BANK["questions"])

    url, (first, second) = seed_into(tmp_path, bank_file, bank_file)

    assert (first, second) == (2, 0)  # повторный старт не дублирует вопросы
    eng = create_engine(url)
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT target_word, distractor_3, is_approved, generation_log FROM questions "
            "ORDER BY target_word")).all()
    eng.dispose()
    assert [r.target_word for r in rows] == ["бежать", "кот"]
    assert rows[0].distractor_3 is None       # два дистрактора — третьего нет
    assert all(r.is_approved for r in rows)
    assert json.loads(rows[1].generation_log) == [{"step": "start", "word": "кот"}]


def test_class_added_to_bank_later_is_seeded(tmp_path):
    """Класс, дописанный в набор после первого деплоя, появляется при
    следующем старте, а уже загруженные классы не дублируются."""
    first_bank = write_bank(tmp_path / "v1.json", BANK["questions"])
    extra = dict(BANK["questions"][0], target_word="стол", correct_answer="стол", word_class=6)
    second_bank = write_bank(tmp_path / "v2.json", BANK["questions"] + [extra])

    _, added = seed_into(tmp_path, first_bank, second_bank)
    assert added == [2, 1]


def test_missing_file_or_disabled_seeding_does_nothing(tmp_path, monkeypatch):
    _, (first,) = seed_into(tmp_path, tmp_path / "нет-такого.json")
    assert first == 0
    monkeypatch.setenv("RULEX_SEED_BANK", "")
    assert seed.seed_path() is None


def test_committed_demo_bank_is_valid():
    """Файл, который загружает демо, читается и годится для теста."""
    if not seed.DEFAULT_BANK.exists():
        return
    bank = json.loads(seed.DEFAULT_BANK.read_text(encoding="utf-8"))
    for item in bank["questions"]:
        question = seed._question(item)
        assert question.definition and question.correct_answer
        assert question.distractor_1 and question.distractor_2
        assert question.frequency_type in {"high", "medium", "low"}
