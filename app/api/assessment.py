"""Прохождение теста: выдача вопросов, проверка ответов, история.

Обработчики тонкие: выбор вопросов, проверка и подсчёт итога — чистые
функции в app.services.assessment.

Тест привязан к попытке (TestAttempt): старт фиксирует набор выданных
вопросов, проверка принимает ответы только на них и только один раз.
Так устроены и тест ученика, и тест без регистрации — у гостя попытка
без владельца, а результат не сохраняется.
"""
import json
import secrets
from datetime import timedelta
from types import SimpleNamespace

from fastapi import APIRouter, HTTPException
from sqlalchemy import and_, delete, func, select

from app.api.deps import CurrentUser
from app.api.schemas import TestCompleteRequest, TestStartRequest
from app.core.database import AsyncSessionLocal
from app.core.models import Question, TestAnswer, TestAttempt, TestResult, utcnow
from app.core.security import hash_token
from app.services.assessment import (
    LEVEL_TEXT,
    MIN_QUESTIONS,
    build_question_payload,
    grade_answers,
    select_test_questions,
    summarize,
)

router = APIRouter(tags=["assessment"])

# Сколько живёт незавершённая попытка. Тест из 20 вопросов проходят
# за минуты; запас — на случай, если вкладку оставили открытой.
ATTEMPT_TTL = timedelta(hours=3)


async def _approved_questions(grade: int) -> list[Question]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Question).where(and_(Question.is_approved.is_(True), Question.word_class == grade))
        )
        return list(result.scalars().all())


async def _start_attempt(questions: list[Question], grade: int, user_id: int | None) -> dict:
    """Выдать тест: выбрать вопросы и запомнить, какие именно выданы."""
    selected = select_test_questions(questions)
    token = secrets.token_urlsafe(24)
    now = utcnow()
    async with AsyncSessionLocal() as session:
        # Попутно убираем брошенные попытки, чтобы таблица не росла
        await session.execute(delete(TestAttempt).where(TestAttempt.expires_at < now))
        session.add(TestAttempt(
            token_hash=hash_token(token), user_id=user_id, word_class=grade,
            question_ids=json.dumps([q.id for q in selected]),
            created_at=now, expires_at=now + ATTEMPT_TTL,
        ))
        await session.commit()

    payload = [build_question_payload(q) for q in selected]
    return {"attempt_id": token, "questions": payload, "grade": grade, "total": len(payload)}


async def _grade_attempt(data: TestCompleteRequest, user_id: int | None) -> tuple[int, list[dict], dict]:
    """Проверить ответы попытки и закрыть её. Возвращает (класс, проверенные ответы, итог).

    Засчитываются только выданные вопросы; вопрос без ответа — неверный,
    поэтому пропуск трудных вопросов процент не поднимает. Попытка
    одноразовая: повторная отправка тех же ответов отклоняется.
    """
    owner = TestAttempt.user_id.is_(None) if user_id is None else TestAttempt.user_id == user_id
    token_hash = hash_token(data.attempt_id)
    not_found = HTTPException(status_code=400, detail="Тест не найден или уже завершён — начните новый")

    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(TestAttempt.word_class, TestAttempt.question_ids)
            .where(TestAttempt.token_hash == token_hash, TestAttempt.expires_at >= utcnow(), owner)
        )).first()
        if row is None:
            raise not_found
        grade, issued = row.word_class, json.loads(row.question_ids)

        foreign = sorted({a.question_id for a in data.answers} - set(issued))
        if foreign:
            raise HTTPException(status_code=400, detail=f"Вопросы не из этого теста: {foreign[:5]}")

        # Удаление — это и есть «занять» попытку: из двух одновременных
        # отправок строку удалит только одна
        claimed = (await session.execute(
            delete(TestAttempt).where(TestAttempt.token_hash == token_hash)
        )).rowcount
        await session.commit()
        if claimed != 1:
            raise not_found

        # Эталоны — из БД. Вопрос, удалённый из банка во время теста, не считается.
        result = await session.execute(select(Question).where(Question.id.in_(issued)))
        questions = {q.id: q for q in result.scalars().all()}

    given = {a.question_id: a.answer for a in data.answers}
    answers = [SimpleNamespace(question_id=qid, answer=given.get(qid))
               for qid in issued if qid in questions]
    if not answers:
        raise HTTPException(status_code=400, detail="В тесте не осталось вопросов")

    graded = grade_answers(questions, answers)
    # В рекомендации — класс ученика: списки слов нумеруются на единицу
    # меньше (ученику 6 класса — список 5 класса, см. /api/available-classes).
    # Раньше ученику 6 класса советовали повторить «слова 5 класса».
    return grade, graded, summarize(graded, grade + 1)


def _result_payload(s: dict, grade: int) -> dict:
    high_correct, high_total, high_pct = s["high"]
    medium_correct, medium_total = s["medium"]
    low_correct, low_total, low_pct = s["low"]
    return {
        "score": s["score"],
        "total": s["total"],
        "percentage": round(s["percentage"], 1),
        "level": s["level"],
        "level_text": LEVEL_TEXT[s["level"]],
        "grade": grade,
        "max_difficulty": s["max_difficulty"],
        "recommendation": s["recommendation"],
        "details": {
            "high_freq": {"correct": high_correct, "total": high_total, "percentage": round(high_pct, 1)},
            "medium_freq": {"correct": medium_correct, "total": medium_total},
            "low_freq": {"correct": low_correct, "total": low_total, "percentage": round(low_pct, 1)},
        },
    }


@router.get("/api/public/grades")
async def available_grades():
    """Сколько одобренных вопросов в банке по каждому классу слов.

    Интерфейс по этому списку показывает только те классы, для которых
    тест можно собрать. Раньше он считал вопросы через /api/questions,
    а тот после разграничения прав доступен только преподавателю:
    у ученика запрос падал с 403, и счётчик молча не обновлялся.
    """
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(Question.word_class, func.count(Question.id))
            .where(Question.is_approved.is_(True), Question.word_class.is_not(None))
            .group_by(Question.word_class)
        )).all()
    return [{"word_class": wc, "questions": n, "ready": n >= MIN_QUESTIONS} for wc, n in sorted(rows)]


@router.post("/api/test/start")
async def start_test(data: TestStartRequest, user: CurrentUser):
    questions = await _approved_questions(data.grade)
    if len(questions) < MIN_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Недостаточно вопросов для {data.grade} класса")
    return await _start_attempt(questions, data.grade, user.id)


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
    return await _start_attempt(questions, data.grade, None)


@router.post("/api/test/complete")
async def complete_test(data: TestCompleteRequest, user: CurrentUser):
    grade, graded, s = await _grade_attempt(data, user.id)
    high_correct, high_total, _ = s["high"]
    medium_correct, medium_total = s["medium"]
    low_correct, low_total, _ = s["low"]

    async with AsyncSessionLocal() as session:
        test_result = TestResult(
            user_id=user.id,
            score=s["score"],
            total_questions=s["total"],
            percentage=s["percentage"],
            high_freq_correct=high_correct, high_freq_total=high_total,
            medium_freq_correct=medium_correct, medium_freq_total=medium_total,
            low_freq_correct=low_correct, low_freq_total=low_total,
            grade_tested=grade,
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

    return _result_payload(s, grade)


@router.post("/api/public/test/complete")
async def complete_public_test(data: TestCompleteRequest):
    """Проверка теста без регистрации: ответы сверяются на сервере,
    результат не сохраняется. Принимается только гостевая попытка,
    выданная /api/public/test/auto-start."""
    grade, _, s = await _grade_attempt(data, None)
    return _result_payload(s, grade)


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
