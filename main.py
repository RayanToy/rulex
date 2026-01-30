# -*- coding: utf-8 -*-
import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from fastapi import FastAPI, Request, HTTPException, Response, Cookie
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select
from pydantic import BaseModel
from typing import List, Optional
import random

from database import init_db, AsyncSessionLocal
from models import Question, User, TestResult
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
    part_of_speech: Optional[str] = None


# Вспомогательная функция для проверки авторизации
async def get_current_user(token: str) -> Optional[User]:
    if not token:
        return None
    user_id = get_user_id_from_token(token)
    if not user_id:
        return None
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()


# Страницы
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ============ AUTH API ============

@app.post("/api/auth/register")
async def register(data: UserRegister):
    async with AsyncSessionLocal() as session:
        # Проверка существования пользователя
        result = await session.execute(
            select(User).where((User.username == data.username) | (User.email == data.email))
        )
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=400, detail="Пользователь уже существует")
        
        # Создание пользователя
        user = User(
            username=data.username,
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        
        # Создание сессии
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
                "correct": correct_index
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)