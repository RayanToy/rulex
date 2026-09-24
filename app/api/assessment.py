"""Прохождение теста: выдача вопросов, проверка ответов, история.

Обработчики тонкие: выбор вопросов, проверка и подсчёт итога — чистые
функции в app.services.assessment.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select

from app.api.deps import CurrentUser, require_user
from app.api.schemas import TestCompleteRequest, TestStartRequest
from app.core.database import AsyncSessionLocal
from app.core.models import Question, TestAnswer, TestResult
from app.services.assessment import (
    LEVEL_TEXT,
    MIN_QUESTIONS,
    build_question_payload,
    grade_answers,
    select_test_questions,
    summarize,
)

router = APIRouter(tags=["assessment"])


async def _approved_questions(grade: int) -> list[Question]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Question).where(and_(Question.is_approved.is_(True), Question.word_class == grade))
        )
        return list(result.scalars().all())


def _test_payload(questions: list[Question], grade: int) -> dict:
    payload = [build_question_payload(q) for q in select_test_questions(questions)]
    return {"questions": payload, "grade": grade, "total": len(payload)}


@router.post("/api/test/start", dependencies=[Depends(require_user)])
async def start_test(data: TestStartRequest):
    questions = await _approved_questions(data.grade)
    if len(questions) < MIN_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Недостаточно вопросов для {data.grade} класса")
    return _test_payload(questions, data.grade)


@router.post("/api/public/test/auto-start")
async def start_public_test(data: TestStartRequest):
    """Тест без регистрации — только по готовому банку вопросов.

    Генерация отсюда убрана намеренно. Эндпоинт открыт без авторизации,
    а генерация 20 вопросов — это десятки платных вызовов LLM: любой
    желающий мог обнулить биллинг простым циклом запросов. Наполняет
    банк вопросов учитель через /api/auto-generate.
    """
    questions = await _approved_questions(data.grade)
    if len(questions) < MIN_QUESTIONS:
        raise HTTPException(
            status_code=503,
            detail=f"Для {data.grade} класса ещё не подготовлены вопросы. "
                   f"Банк вопросов наполняет преподаватель.",
        )
    return _test_payload(questions, data.grade)


@router.post("/api/test/complete")
async def complete_test(data: TestCompleteRequest, user: CurrentUser):
    if not data.answers:
        raise HTTPException(status_code=400, detail="Нет ответов")

    # Эталоны — из БД; клиенту не доверяется ничего, кроме выбранного варианта
    question_ids = [a.question_id for a in data.answers]
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Question).where(Question.id.in_(question_ids)))
        questions = {q.id: q for q in result.scalars().all()}

    missing = [qid for qid in question_ids if qid not in questions]
    if missing:
        raise HTTPException(status_code=400, detail=f"Неизвестные вопросы: {missing[:5]}")

    graded = grade_answers(questions, data.answers)
    s = summarize(graded, data.grade)
    high_correct, high_total, high_pct = s["high"]
    medium_correct, medium_total = s["medium"]
    low_correct, low_total, low_pct = s["low"]

    async with AsyncSessionLocal() as session:
        test_result = TestResult(
            user_id=user.id,
            score=s["score"],
            total_questions=s["total"],
            percentage=s["percentage"],
            high_freq_correct=high_correct, high_freq_total=high_total,
            medium_freq_correct=medium_correct, medium_freq_total=medium_total,
            low_freq_correct=low_correct, low_freq_total=low_total,
            grade_tested=data.grade,
            level_achieved=s["level"],
            max_difficulty_reached=s["max_difficulty"],
            recommendation=s["recommendation"],
        )
        session.add(test_result)
        await session.flush()  # нужен id результата для ответов
        session.add_all([
            TestAnswer(
                test_result_id=test_result.id,
                question_id=a["question_id"],
                is_correct=a["is_correct"],
                user_answer=a["user_answer"],
                correct_answer=a["correct_answer"],
                difficulty_at_answer=a["difficulty"],
            )
            for a in graded
        ])
        await session.commit()

    return {
        "score": s["score"],
        "total": s["total"],
        "percentage": round(s["percentage"], 1),
        "level": s["level"],
        "level_text": LEVEL_TEXT[s["level"]],
        "grade": data.grade,
        "max_difficulty": s["max_difficulty"],
        "recommendation": s["recommendation"],
        "details": {
            "high_freq": {"correct": high_correct, "total": high_total, "percentage": round(high_pct, 1)},
            "medium_freq": {"correct": medium_correct, "total": medium_total},
            "low_freq": {"correct": low_correct, "total": low_total, "percentage": round(low_pct, 1)},
        },
    }


@router.get("/api/test/history")
async def history(user: CurrentUser):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TestResult).where(TestResult.user_id == user.id)
            .order_by(TestResult.completed_at.desc())
        )
        return [{
            "id": r.id,
            "score": r.score,
            "total": r.total_questions,
            "percentage": r.percentage,
            "grade": r.grade_tested,
            "level": r.level_achieved,
            "recommendation": r.recommendation,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        } for r in result.scalars().all()]
