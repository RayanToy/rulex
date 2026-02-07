# -*- coding: utf-8 -*-
import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from fastapi import FastAPI, Request, HTTPException, Cookie
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select, and_
from pydantic import BaseModel
from typing import List, Optional
import random

from database import init_db, AsyncSessionLocal
from models import Question, User, TestResult, TestAnswer
from auth import hash_password, verify_password, create_session, get_user_id_from_token, delete_session

app = FastAPI(title="RuLex")


@app.on_event("startup")
async def startup():
    init_db()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Pydantic модели
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    grade: Optional[int] = 6


class UserLogin(BaseModel):
    username: str
    password: str


class WordInput(BaseModel):
    word: str


class QuestionCreate(BaseModel):
    target_word: str
    definition: str
    correct_answer: str
    distractor_1: Optional[str] = None
    distractor_2: Optional[str] = None
    distractor_3: Optional[str] = None
    word_class: int = 6
    frequency_type: str = "medium"
    difficulty: int = 5
    part_of_speech: Optional[str] = None


class TestStartRequest(BaseModel):
    grade: int = 6


class TestAnswerRequest(BaseModel):
    question_id: int
    answer: str
    is_correct: bool


class TestCompleteRequest(BaseModel):
    answers: List[dict]
    grade: int


# Вспомогательные функции
async def get_current_user(token: str) -> Optional[User]:
    if not token:
        return None
    user_id = get_user_id_from_token(token)
    if not user_id:
        return None
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()


def calculate_level(percentage: float, high_freq_pct: float, low_freq_pct: float) -> str:
    """Определение уровня по результатам"""
    if percentage >= 90 and low_freq_pct >= 70:
        return "high"
    elif percentage >= 70:
        return "medium"
    else:
        return "low"


def get_recommendation(level: str, grade: int, percentage: float) -> str:
    """Генерация рекомендации"""
    if level == "high":
        return f"Отлично! Вы показали высокий уровень владения лексикой ({percentage:.0f}%). Рекомендуем перейти к изучению слов {grade + 1} класса."
    elif level == "medium":
        return f"Хороший результат ({percentage:.0f}%). Вы знаете большинство слов {grade} класса. Рекомендуем уделить внимание редким словам."
    else:
        return f"Результат: {percentage:.0f}%. Рекомендуем повторить основные слова {grade} класса."

# Страницы
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ============ AUTH API ============

@app.post("/api/auth/register")
async def register(data: UserRegister):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where((User.username == data.username) | (User.email == data.email))
        )
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=400, detail="Пользователь уже существует")
        
        user = User(
            username=data.username,
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
            grade=data.grade or 6
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        
        token = create_session(user.id)
        
        response = JSONResponse(content={"message": "OK", "user": user.to_dict()})
        response.set_cookie(key="session_token", value=token, httponly=True, max_age=604800)
        return response


@app.post("/api/auth/login")
async def login(data: UserLogin):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.username == data.username))
        user = result.scalar_one_or_none()
        
        if not user or not verify_password(data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Неверный логин или пароль")
        
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Аккаунт заблокирован")
        
        token = create_session(user.id)
        
        response = JSONResponse(content={"message": "OK", "user": user.to_dict()})
        response.set_cookie(key="session_token", value=token, httponly=True, max_age=604800)
        return response


@app.post("/api/auth/logout")
async def logout(session_token: Optional[str] = Cookie(None)):
    if session_token:
        delete_session(session_token)
    response = JSONResponse(content={"message": "OK"})
    response.delete_cookie("session_token")
    return response


@app.get("/api/auth/me")
async def get_me(session_token: Optional[str] = Cookie(None)):
    if not session_token:
        raise HTTPException(status_code=401, detail="Не авторизован")
    
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Не авторизован")
    
    return user.to_dict()


# ============ ADAPTIVE TEST API ============

@app.post("/api/test/start")
async def start_adaptive_test(data: TestStartRequest, session_token: Optional[str] = Cookie(None)):
    """Начать адаптивный тест"""
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    grade = data.grade
    
    async with AsyncSessionLocal() as session:
        # Получаем вопросы для класса
        result = await session.execute(
            select(Question).where(
                and_(Question.is_approved == True, Question.word_class == grade)
            )
        )
        all_questions = result.scalars().all()
        
        if len(all_questions) < 5:
            raise HTTPException(status_code=400, detail=f"Недостаточно вопросов для {grade} класса")
        
        # Разделяем по частотности
        high_freq = [q for q in all_questions if q.frequency_type == "high"]
        medium_freq = [q for q in all_questions if q.frequency_type == "medium"]
        low_freq = [q for q in all_questions if q.frequency_type == "low"]
        
        # Если нет разделения, используем все как medium
        if not high_freq and not medium_freq and not low_freq:
            medium_freq = all_questions
        
        # 70% высоко/средне частотные, 30% низкочастотные
        target_total = min(20, len(all_questions))
        target_low = max(1, int(target_total * 0.3))
        target_high_medium = target_total - target_low
        
        selected = []
        
        # Выбираем высокочастотные и среднечастотные
        high_medium_pool = high_freq + medium_freq
        if high_medium_pool:
            selected.extend(random.sample(high_medium_pool, min(target_high_medium, len(high_medium_pool))))
        
        # Выбираем низкочастотные
        if low_freq:
            selected.extend(random.sample(low_freq, min(target_low, len(low_freq))))
        
        # Если не хватает, добираем из общего пула
        if len(selected) < target_total:
            remaining = [q for q in all_questions if q not in selected]
            need = target_total - len(selected)
            selected.extend(random.sample(remaining, min(need, len(remaining))))
        
        # Перемешиваем
        random.shuffle(selected)
        
        # Форматируем для фронтенда
        questions_data = []
        for q in selected:
            options = [q.correct_answer]
            if q.distractor_1:
                options.append(q.distractor_1)
            if q.distractor_2:
                options.append(q.distractor_2)
            if q.distractor_3:
                options.append(q.distractor_3)
            
            random.shuffle(options)
            correct_index = options.index(q.correct_answer)
            
            questions_data.append({
                "id": q.id,
                "question": q.definition,
                "options": options,
                "correct": correct_index,
                "target_word": q.target_word,
                "frequency_type": q.frequency_type or "medium",
                "difficulty": q.difficulty or 5
            })
        
        return {
            "questions": questions_data,
            "grade": grade,
            "total": len(questions_data)
        }


@app.post("/api/test/complete")
async def complete_adaptive_test(data: TestCompleteRequest, session_token: Optional[str] = Cookie(None)):
    """Завершить тест и получить результаты"""
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    answers = data.answers
    grade = data.grade
    
    # Подсчёт результатов
    total = len(answers)
    correct = sum(1 for a in answers if a.get("is_correct"))
    percentage = (correct / total * 100) if total > 0 else 0
    
    # Подсчёт по частотности
    high_correct = sum(1 for a in answers if a.get("is_correct") and a.get("frequency_type") == "high")
    high_total = sum(1 for a in answers if a.get("frequency_type") == "high")
    
    medium_correct = sum(1 for a in answers if a.get("is_correct") and a.get("frequency_type") == "medium")
    medium_total = sum(1 for a in answers if a.get("frequency_type") == "medium")
    
    low_correct = sum(1 for a in answers if a.get("is_correct") and a.get("frequency_type") == "low")
    low_total = sum(1 for a in answers if a.get("frequency_type") == "low")
    
    # Проценты
    high_pct = (high_correct / high_total * 100) if high_total > 0 else 100
    low_pct = (low_correct / low_total * 100) if low_total > 0 else 0
    
    # Максимальная достигнутая сложность (адаптивная логика)
    max_difficulty = 5
    consecutive_correct = 0
    consecutive_wrong = 0
    current_difficulty = 5
    
    for a in answers:
        if a.get("is_correct"):
            consecutive_correct += 1
            consecutive_wrong = 0
            if consecutive_correct >= 3:
                current_difficulty = min(10, current_difficulty + 1)
                consecutive_correct = 0
        else:
            consecutive_wrong += 1
            consecutive_correct = 0
            if consecutive_wrong >= 2:
                current_difficulty = max(1, current_difficulty - 1)
                consecutive_wrong = 0
        
        max_difficulty = max(max_difficulty, current_difficulty)
    
    # Определяем уровень
    level = calculate_level(percentage, high_pct, low_pct)
    recommendation = get_recommendation(level, grade, percentage)
    
    # Сохраняем результат
    async with AsyncSessionLocal() as session:
        test_result = TestResult(
            user_id=user.id,
            score=correct,
            total_questions=total,
            percentage=percentage,
            high_freq_correct=high_correct,
            high_freq_total=high_total,
            medium_freq_correct=medium_correct,
            medium_freq_total=medium_total,
            low_freq_correct=low_correct,
            low_freq_total=low_total,
            grade_tested=grade,
            level_achieved=level,
            max_difficulty_reached=max_difficulty,
            recommendation=recommendation
        )
        session.add(test_result)
        await session.commit()
        await session.refresh(test_result)
        
        # Сохраняем отдельные ответы
        for a in answers:
            answer = TestAnswer(
                test_result_id=test_result.id,
                question_id=a.get("question_id", 0),
                is_correct=a.get("is_correct", False),
                user_answer=a.get("user_answer"),
                correct_answer=a.get("correct_answer"),
                difficulty_at_answer=a.get("difficulty", 5)
            )
            session.add(answer)
        
        await session.commit()
    
    return {
        "score": correct,
        "total": total,
        "percentage": round(percentage, 1),
        "level": level,
        "level_text": {"high": "Высокий", "medium": "Средний", "low": "Низкий"}[level],
        "grade": grade,
        "max_difficulty": max_difficulty,
        "recommendation": recommendation,
        "details": {
            "high_freq": {"correct": high_correct, "total": high_total, "percentage": round(high_pct, 1)},
            "medium_freq": {"correct": medium_correct, "total": medium_total},
            "low_freq": {"correct": low_correct, "total": low_total, "percentage": round(low_pct, 1)}
        }
    }


@app.get("/api/test/history")
async def get_test_history(session_token: Optional[str] = Cookie(None)):
    """История тестирований пользователя"""
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TestResult).where(TestResult.user_id == user.id).order_by(TestResult.completed_at.desc())
        )
        results = result.scalars().all()
        
        return [{
            "id": r.id,
            "score": r.score,
            "total": r.total_questions,
            "percentage": r.percentage,
            "grade": r.grade_tested,
            "level": r.level_achieved,
            "recommendation": r.recommendation,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None
        } for r in results]


# ============ QUESTIONS API ============

@app.post("/api/generate-and-save")
async def generate_and_save(data: WordInput, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    try:
        from generator import QuestionGenerator
        generator = QuestionGenerator()
        result = generator.generate_question(data.word)
        
        async with AsyncSessionLocal() as session:
            db_question = Question(
                target_word=result["target_word"],
                definition=result["definition"],
                correct_answer=result["correct_answer"],
                distractor_1=result["distractors"][0] if len(result["distractors"]) > 0 else None,
                distractor_2=result["distractors"][1] if len(result["distractors"]) > 1 else None,
                distractor_3=result["distractors"][2] if len(result["distractors"]) > 2 else None,
                part_of_speech=result.get("part_of_speech"),
                word_class=result.get("word_class", 6),
                frequency_type="medium",
                difficulty=5,
                generation_log=result.get("generation_log"),
                is_approved=True,
                created_by=user.id
            )
            session.add(db_question)
            await session.commit()
            await session.refresh(db_question)
            
            return {"id": db_question.id, "question": db_question.to_dict(), "message": "OK"}
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/questions")
async def create_question(question: QuestionCreate, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    async with AsyncSessionLocal() as session:
        db_question = Question(
            target_word=question.target_word,
            definition=question.definition,
            correct_answer=question.correct_answer,
            distractor_1=question.distractor_1,
            distractor_2=question.distractor_2,
            distractor_3=question.distractor_3,
            word_class=question.word_class,
            frequency_type=question.frequency_type,
            difficulty=question.difficulty,
            part_of_speech=question.part_of_speech,
            is_approved=True,
            created_by=user.id
        )
        session.add(db_question)
        await session.commit()
        await session.refresh(db_question)
        return {"id": db_question.id, "message": "OK"}


@app.get("/api/questions")
async def get_all_questions(session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Question).where(Question.is_approved == True))
        questions = result.scalars().all()
        return [q.to_dict() for q in questions]


@app.get("/api/questions/full")
async def get_all_questions_full(session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Question).where(Question.is_approved == True))
        questions = result.scalars().all()
        
        full_data = []
        for q in questions:
            options = [q.correct_answer]
            if q.distractor_1:
                options.append(q.distractor_1)
            if q.distractor_2:
                options.append(q.distractor_2)
            if q.distractor_3:
                options.append(q.distractor_3)
            
            full_data.append({
                "id": q.id,
                "target_word": q.target_word,
                "question": q.definition,
                "correct_answer": q.correct_answer,
                "distractor_1": q.distractor_1,
                "distractor_2": q.distractor_2,
                "distractor_3": q.distractor_3,
                "options": options,
                "correct": 0,
                "word_class": q.word_class,
                "frequency_type": q.frequency_type,
                "difficulty": q.difficulty,
                "part_of_speech": q.part_of_speech
            })
        
        return full_data


@app.put("/api/questions/{question_id}")
async def update_question(question_id: int, question: QuestionCreate, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Question).where(Question.id == question_id))
        db_question = result.scalar_one_or_none()
        if not db_question:
            raise HTTPException(status_code=404, detail="Not found")
        
        db_question.target_word = question.target_word
        db_question.definition = question.definition
        db_question.correct_answer = question.correct_answer
        db_question.distractor_1 = question.distractor_1
        db_question.distractor_2 = question.distractor_2
        db_question.distractor_3 = question.distractor_3
        db_question.word_class = question.word_class
        db_question.frequency_type = question.frequency_type
        db_question.difficulty = question.difficulty
        if question.part_of_speech:
            db_question.part_of_speech = question.part_of_speech
        
        await session.commit()
        return {"message": "OK"}


@app.get("/api/questions/random")
async def get_random_questions(count: int = 20, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Question).where(Question.is_approved == True))
        questions = result.scalars().all()
        
        if len(questions) < count:
            selected = questions
        else:
            selected = random.sample(list(questions), count)
        
        formatted = []
        for q in selected:
            options = [q.correct_answer]
            if q.distractor_1:
                options.append(q.distractor_1)
            if q.distractor_2:
                options.append(q.distractor_2)
            if q.distractor_3:
                options.append(q.distractor_3)
            
            random.shuffle(options)
            correct_index = options.index(q.correct_answer)
            
            formatted.append({
                "id": q.id,
                "question": q.definition,
                "options": options,
                "correct": correct_index,
                "frequency_type": q.frequency_type or "medium",
                "difficulty": q.difficulty or 5
            })
        
        return formatted


@app.delete("/api/questions/{question_id}")
async def delete_question(question_id: int, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Question).where(Question.id == question_id))
        question = result.scalar_one_or_none()
        if not question:
            raise HTTPException(status_code=404, detail="Not found")
        await session.delete(question)
        await session.commit()
        return {"message": "OK"}


@app.post("/api/auto-generate")
async def auto_generate_questions(data: TestStartRequest, session_token: Optional[str] = Cookie(None)):
    """Автоматическая генерация вопросов для класса из списка слов"""
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    grade = data.grade
    
    try:
        from generator import QuestionGenerator
        generator = QuestionGenerator()
        
        if not generator.has_word_list(grade):
            raise HTTPException(status_code=400, detail=f"Нет списка слов для {grade} класса")
        
        # Генерируем 20 вопросов
        questions = generator.generate_questions_for_class(grade, 20)
        
        if not questions:
            raise HTTPException(status_code=500, detail="Не удалось сгенерировать вопросы")
        
        # Сохраняем в БД
        saved_count = 0
        async with AsyncSessionLocal() as session:
            for q in questions:
                db_question = Question(
                    target_word=q["target_word"],
                    definition=q["definition"],
                    correct_answer=q["correct_answer"],
                    distractor_1=q["distractors"][0] if len(q["distractors"]) > 0 else None,
                    distractor_2=q["distractors"][1] if len(q["distractors"]) > 1 else None,
                    distractor_3=q["distractors"][2] if len(q["distractors"]) > 2 else None,
                    part_of_speech=q.get("part_of_speech"),
                    word_class=grade,
                    frequency_type=q.get("frequency_type", "medium"),
                    difficulty=5,
                    is_approved=True,
                    created_by=user.id
                )
                session.add(db_question)
                saved_count += 1
            
            await session.commit()
        
        return {"message": f"Сгенерировано и сохранено {saved_count} вопросов для {grade} класса"}
    
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/public/test/auto-start")
async def auto_start_public_test(data: TestStartRequest):
    """Начать тест с автогенерацией если вопросов нет"""
    grade = data.grade
    
    async with AsyncSessionLocal() as session:
        # Проверяем есть ли вопросы
        result = await session.execute(
            select(Question).where(
                and_(Question.is_approved == True, Question.word_class == grade)
            )
        )
        all_questions = result.scalars().all()
        
        # Если вопросов мало - генерируем автоматически
        if len(all_questions) < 5:
            try:
                from generator import QuestionGenerator
                generator = QuestionGenerator()
                
                if generator.has_word_list(grade):
                    # Генерируем 20 вопросов
                    new_questions = generator.generate_questions_for_class(grade, 20)
                    
                    for q in new_questions:
                        db_question = Question(
                            target_word=q["target_word"],
                            definition=q["definition"],
                            correct_answer=q["correct_answer"],
                            distractor_1=q["distractors"][0] if len(q["distractors"]) > 0 else None,
                            distractor_2=q["distractors"][1] if len(q["distractors"]) > 1 else None,
                            distractor_3=q["distractors"][2] if len(q["distractors"]) > 2 else None,
                            part_of_speech=q.get("part_of_speech"),
                            word_class=grade,
                            frequency_type=q.get("frequency_type", "medium"),
                            difficulty=5,
                            is_approved=True
                        )
                        session.add(db_question)
                    
                    await session.commit()
                    
                    # Перезагружаем вопросы
                    result = await session.execute(
                        select(Question).where(
                            and_(Question.is_approved == True, Question.word_class == grade)
                        )
                    )
                    all_questions = result.scalars().all()
                else:
                    raise HTTPException(status_code=400, detail=f"Нет вопросов и списка слов для {grade} класса")
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")
        
        if len(all_questions) < 5:
            raise HTTPException(status_code=400, detail=f"Недостаточно вопросов для {grade} класса")
        
        # Дальше обычная логика выбора вопросов
        high_freq = [q for q in all_questions if q.frequency_type == "high"]
        medium_freq = [q for q in all_questions if q.frequency_type == "medium"]
        low_freq = [q for q in all_questions if q.frequency_type == "low"]
        
        if not high_freq and not medium_freq and not low_freq:
            medium_freq = all_questions
        
        target_total = min(20, len(all_questions))
        target_low = max(1, int(target_total * 0.3))
        target_high_medium = target_total - target_low
        
        selected = []
        
        high_medium_pool = high_freq + medium_freq
        if high_medium_pool:
            selected.extend(random.sample(high_medium_pool, min(target_high_medium, len(high_medium_pool))))
        
        if low_freq:
            selected.extend(random.sample(low_freq, min(target_low, len(low_freq))))
        
        if len(selected) < target_total:
            remaining = [q for q in all_questions if q not in selected]
            need = target_total - len(selected)
            selected.extend(random.sample(remaining, min(need, len(remaining))))
        
        random.shuffle(selected)
        
        questions_data = []
        for q in selected:
            options = [q.correct_answer]
            if q.distractor_1:
                options.append(q.distractor_1)
            if q.distractor_2:
                options.append(q.distractor_2)
            if q.distractor_3:
                options.append(q.distractor_3)
            
            random.shuffle(options)
            correct_index = options.index(q.correct_answer)
            
            questions_data.append({
                "id": q.id,
                "question": q.definition,
                "options": options,
                "correct": correct_index,
                "target_word": q.target_word,
                "frequency_type": q.frequency_type or "medium",
                "difficulty": q.difficulty or 5
            })
        
        return {
            "questions": questions_data,
            "grade": grade,
            "total": len(questions_data),
            "auto_generated": len(all_questions) < 5
        }


@app.get("/api/available-classes")
async def get_available_classes():
    """Получить список классов с доступными словами"""
    try:
        from generator import QuestionGenerator
        generator = QuestionGenerator()
        classes = generator.get_available_classes()
        
        # Формируем информацию
        # word_class N = для ученика класса N+1
        result = []
        for word_class in classes:
            student_class = word_class + 1
            if student_class <= 11:
                label = f"{student_class} класс"
            else:
                label = "Выпускник"
            
            result.append({
                "word_class": word_class,
                "student_class": student_class,
                "label": label
            })
        
        return result
    except Exception as e:
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)