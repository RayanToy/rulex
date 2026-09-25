"""Банк вопросов и генерация — учительская часть.

Все эндпоинты, кроме списка классов, требуют прав администратора.
"""
import asyncio
import logging
import random

from anthropic import APIConnectionError, APIError, APIStatusError
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from app.api.deps import AdminUser, require_admin
from app.api.schemas import QuestionCreate, TestStartRequest, WordInput
from app.core.database import AsyncSessionLocal
from app.core.models import Question
from app.services.assessment import question_options
from app.services.generation import QuestionGenerator
from app.services.llm import OllamaError
from app.services.wordlists import get_word_manager

log = logging.getLogger(__name__)
router = APIRouter(tags=["questions"])

AUTO_GENERATE_COUNT = 20


def _question_from_generated(q: dict, word_class: int, user_id: int, frequency_type: str) -> Question:
    distractors = q.get("distractors") or []
    return Question(
        target_word=q["target_word"],
        definition=q["definition"],
        correct_answer=q["correct_answer"],
        distractor_1=distractors[0] if len(distractors) > 0 else None,
        distractor_2=distractors[1] if len(distractors) > 1 else None,
        distractor_3=distractors[2] if len(distractors) > 2 else None,
        part_of_speech=q.get("part_of_speech"),
        word_class=word_class,
        frequency_type=frequency_type,
        difficulty=5,
        generation_log=q.get("generation_log"),
        is_approved=True,
        created_by=user_id,
    )


def _generation_error(exc: Exception) -> HTTPException:
    """Раньше любой сбой генерации был 500. Различаем причины: слово не
    подошло — это не ошибка сервера, а сбой модели — ошибка вышестоящего
    сервиса."""
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (APIStatusError, APIConnectionError, APIError, OllamaError)):
        return HTTPException(status_code=502, detail=f"Сбой модели: {type(exc).__name__}")
    log.exception("генерация упала")
    return HTTPException(status_code=500, detail="Внутренняя ошибка генерации")


@router.post("/api/generate-and-save")
async def generate_and_save(data: WordInput, user: AdminUser):
    try:
        # Синхронный вызов модели — в потоке, чтобы не блокировать event loop
        result = await asyncio.to_thread(QuestionGenerator().generate_question, data.word)
    except Exception as exc:
        raise _generation_error(exc) from exc

    async with AsyncSessionLocal() as session:
        question = _question_from_generated(result, result.get("word_class", 6), user.id, "medium")
        session.add(question)
        await session.commit()
        await session.refresh(question)
        return {"id": question.id, "question": question.to_dict(), "message": "OK"}


@router.post("/api/auto-generate")
async def auto_generate(data: TestStartRequest, user: AdminUser):
    generator = QuestionGenerator()
    if not generator.has_word_list(data.grade):
        raise HTTPException(status_code=400, detail=f"Нет списка слов для {data.grade} класса")

    try:
        # Асинхронная версия: вызовы модели идут вне event loop,
        # сервер отвечает остальным клиентам, пока идёт генерация.
        generated = await generator.agenerate_questions_for_class(data.grade, AUTO_GENERATE_COUNT)
    except Exception as exc:
        raise _generation_error(exc) from exc
    if not generated:
        raise HTTPException(status_code=502, detail="Не удалось сгенерировать ни одного вопроса")

    async with AsyncSessionLocal() as session:
        session.add_all([
            _question_from_generated(q, data.grade, user.id, q.get("frequency_type", "medium"))
            for q in generated
        ])
        await session.commit()

    saved = len(generated)
    return {
        "message": f"Сгенерировано и сохранено {saved} вопросов для {data.grade} класса",
        "requested": AUTO_GENERATE_COUNT,
        "generated": saved,
        "warning": (f"Удалось сгенерировать только {saved}/{AUTO_GENERATE_COUNT} вопросов. "
                    "Проверьте качество словаря для этого класса.")
        if saved < AUTO_GENERATE_COUNT else None,
    }


@router.post("/api/questions")
async def create_question(question: QuestionCreate, user: AdminUser):
    async with AsyncSessionLocal() as session:
        db_question = Question(**question.model_dump(), is_approved=True, created_by=user.id)
        session.add(db_question)
        await session.commit()
        await session.refresh(db_question)
        return {"id": db_question.id, "message": "OK"}


@router.get("/api/questions", dependencies=[Depends(require_admin)])
async def list_questions():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Question).where(Question.is_approved.is_(True)))
        return [q.to_dict() for q in result.scalars().all()]


@router.get("/api/questions/full", dependencies=[Depends(require_admin)])
async def list_questions_full():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Question).where(Question.is_approved.is_(True)))
        return [{
            "id": q.id,
            "target_word": q.target_word,
            "question": q.definition,
            "correct_answer": q.correct_answer,
            "distractor_1": q.distractor_1,
            "distractor_2": q.distractor_2,
            "distractor_3": q.distractor_3,
            "options": question_options(q),
            "correct": 0,
            "word_class": q.word_class,
            "frequency_type": q.frequency_type,
            "difficulty": q.difficulty,
            "part_of_speech": q.part_of_speech,
        } for q in result.scalars().all()]


@router.put("/api/questions/{question_id}", dependencies=[Depends(require_admin)])
async def update_question(question_id: int, question: QuestionCreate):
    async with AsyncSessionLocal() as session:
        db_question = (await session.execute(
            select(Question).where(Question.id == question_id)
        )).scalar_one_or_none()
        if not db_question:
            raise HTTPException(status_code=404, detail="Not found")

        fields = question.model_dump()
        # part_of_speech не затирается пустым значением — как и раньше
        if not fields.get("part_of_speech"):
            fields.pop("part_of_speech")
        for name, value in fields.items():
            setattr(db_question, name, value)
        await session.commit()
        return {"message": "OK"}


@router.get("/api/questions/random", dependencies=[Depends(require_admin)])
async def random_questions(count: int = 20):
    async with AsyncSessionLocal() as session:
        questions = list((await session.execute(
            select(Question).where(Question.is_approved.is_(True))
        )).scalars().all())

    selected = questions if len(questions) < count else random.sample(questions, count)
    formatted = []
    for q in selected:
        options = question_options(q)
        random.shuffle(options)
        formatted.append({
            "id": q.id,
            "question": q.definition,
            "options": options,
            "correct": options.index(q.correct_answer),  # эндпоинт только для преподавателя
            "frequency_type": q.frequency_type or "medium",
            "difficulty": q.difficulty or 5,
        })
    return formatted


@router.delete("/api/questions/{question_id}", dependencies=[Depends(require_admin)])
async def delete_question(question_id: int):
    async with AsyncSessionLocal() as session:
        question = (await session.execute(
            select(Question).where(Question.id == question_id)
        )).scalar_one_or_none()
        if not question:
            raise HTTPException(status_code=404, detail="Not found")
        # Ответы учеников на этот вопрос не удаляются: внешний ключ
        # ON DELETE SET NULL обнуляет ссылку, история сохраняется.
        await session.delete(question)
        await session.commit()
        return {"message": "OK"}


@router.get("/api/available-classes")
async def available_classes():
    """Классы, для которых есть списки слов.

    Раньше эндпоинт создавал QuestionGenerator ради одного списка ключей,
    а тот при создании поднимает клиента модели и без API-ключа падает.
    Падение глоталось через except -> [], и выбор класса в интерфейсе был
    пустым всякий раз, когда нет ключа или недоступен шлюз. Списку классов
    модель не нужна — только словари.
    """
    manager = await asyncio.to_thread(get_word_manager)
    result = []
    # word_class N — слова для ученика класса N+1
    for word_class in sorted(manager.relative_lists):
        student_class = word_class + 1
        result.append({
            "word_class": word_class,
            "student_class": student_class,
            "label": f"{student_class} класс" if student_class <= 11 else "Выпускник",
        })
    return result
