"""Регистрация, вход, выход."""
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.api.deps import SESSION_COOKIE, CurrentUser, SessionToken, set_session_cookie
from app.api.schemas import UserLogin, UserRegister
from app.core.database import AsyncSessionLocal
from app.core.models import User
from app.core.security import (
    create_session,
    delete_session,
    hash_password,
    needs_rehash,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Становится ли первый зарегистрированный администратором. Удобно
# локально, но на публичном экземпляре им стал бы случайный посетитель,
# поэтому в Docker-образе это выключено (RULEX_FIRST_USER_ADMIN=0).
FIRST_USER_ADMIN = (os.getenv("RULEX_FIRST_USER_ADMIN") or "1").strip().lower() not in {"0", "false", "no"}


@router.post("/register")
async def register(data: UserRegister):
    async with AsyncSessionLocal() as session:
        existing = (await session.execute(
            select(User).where((User.username == data.username) | (User.email == data.email))
        )).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=400, detail="Пользователь уже существует")

        # Первый зарегистрировавшийся становится администратором:
        # иначе учительские эндпоинты недоступны никому и вопросы
        # нечем наполнять. Дальнейших админов назначают через
        # scripts/make_admin.py.
        first_user = FIRST_USER_ADMIN and (
            (await session.execute(select(User.id).limit(1))).first() is None)

        user = User(
            username=data.username,
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
            grade=data.grade or 6,
            is_admin=first_user,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

    token = await create_session(user.id)
    return set_session_cookie(JSONResponse({"message": "OK", "user": user.to_dict()}), token)


@router.post("/login")
async def login(data: UserLogin):
    async with AsyncSessionLocal() as session:
        user = (await session.execute(
            select(User).where(User.username == data.username)
        )).scalar_one_or_none()

        if not user or not verify_password(data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Неверный логин или пароль")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Аккаунт заблокирован")

        # Пароль верный — переводим старый SHA-256-хеш на Argon2 незаметно
        # для пользователя, не требуя смены пароля.
        if needs_rehash(user.hashed_password):
            user.hashed_password = hash_password(data.password)
            await session.commit()

    token = await create_session(user.id)
    return set_session_cookie(JSONResponse({"message": "OK", "user": user.to_dict()}), token)


@router.post("/logout")
async def logout(session_token: SessionToken = None):
    if session_token:
        await delete_session(session_token)
    response = JSONResponse({"message": "OK"})
    response.delete_cookie(SESSION_COOKIE)
    return response


@router.get("/me")
async def me(user: CurrentUser):
    return user.to_dict()
