"""Наполнение банка вопросов готовым набором.

Демо-экземпляр работает без ключа модели: банк вопросов сгенерирован
заранее тем же пайплайном (scripts/build_demo_bank.py) и лежит
в data/demo_questions.json. При старте из набора загружаются классы,
по которым в базе ещё нет ни одного вопроса. Поэтому вопросы,
добавленные преподавателем, не трогаются, а класс, дописанный в набор
позже, появится на уже развёрнутом экземпляре после передеплоя.

RULEX_SEED_BANK задаёт другой файл; пустое значение отключает загрузку.
"""
import json
import os
from pathlib import Path

from sqlalchemy import select

from app import PROJECT_ROOT
from app.core.database import AsyncSessionLocal
from app.core.models import Question

DEFAULT_BANK = PROJECT_ROOT / "data" / "demo_questions.json"


def seed_path() -> Path | None:
    value = os.getenv("RULEX_SEED_BANK")
    if value is None:
        return DEFAULT_BANK
    return Path(value) if value.strip() else None


def _question(item: dict) -> Question:
    distractors = list(item.get("distractors") or []) + [None, None, None]
    log = item.get("generation_log")
    return Question(
        target_word=item["target_word"],
        definition=item["definition"],
        correct_answer=item["correct_answer"],
        distractor_1=distractors[0],
        distractor_2=distractors[1],
        distractor_3=distractors[2],
        part_of_speech=item.get("part_of_speech"),
        word_class=item["word_class"],
        frequency_type=item.get("frequency_type") or "medium",
        difficulty=item.get("difficulty") or 5,
        generation_log=json.dumps(log, ensure_ascii=False) if log is not None else None,
        is_approved=True,
    )


async def seed_question_bank(path: Path | None, session_factory=AsyncSessionLocal) -> int:
    """Загрузить классы набора, которых нет в базе. Возвращает число добавленных вопросов."""
    if path is None or not path.exists():
        return 0
    items = json.loads(path.read_text(encoding="utf-8"))["questions"]
    async with session_factory() as session:
        present = set((await session.execute(select(Question.word_class).distinct())).scalars())
        new = [item for item in items if item["word_class"] not in present]
        if new:
            session.add_all([_question(item) for item in new])
            await session.commit()
    return len(new)
