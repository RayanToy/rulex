"""Зависимости HTTP-слоя: текущий пользователь, проверка прав, кука сессии.

Проверки прав — зависимости FastAPI (Depends), а не ручной вызов в каждом
обработчике: забыть проверку на новом эндпоинте теперь заметнее, а сами
обработчики не занимаются разбором куки.
"""
import os
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.core.models import User
from app.core.security import get_user_id_from_token

SESSION_COOKIE = "session_token"
SESSION_MAX_AGE = 7 * 24 * 3600

# Ставить ли флаг Secure на куку сессии. По HTTP такая кука не отправляется,
# поэтому локальная разработка ломалась бы при значении по умолчанию True.
# В проде это обязано быть включено: RULEX_COOKIE_SECURE=1.
COOKIE_SECURE = (os.getenv("RULEX_COOKIE_SECURE") or "").strip().lower() in {"1", "true", "yes"}

# Токен сессии из куки; None, если куки нет
SessionToken = Annotated[str | None, Cookie(alias=SESSION_COOKIE)]


def set_session_cookie(response: JSONResponse, token: str) -> JSONResponse:
    """SameSite=Lax закрывает базовый CSRF: кука не уходит при
    межсайтовых POST-запросах, а аутентификация тут именно на куке."""
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
        max_age=SESSION_MAX_AGE,
    )
    return response


async def get_current_user(token: str | None) -> User | None:
    if not token:
        return None
    user_id = await get_user_id_from_token(token)
    if not user_id:
        return None
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()


async def require_user(session_token: SessionToken = None) -> User:
    """Любой авторизованный пользователь."""
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    return user


async def require_admin(user: Annotated[User, Depends(require_user)]) -> User:
    """Учительские операции: банк вопросов и генерация.

    Поле is_admin существовало в модели с самого начала, но не
    проверялось нигде — любой зарегистрированный ученик мог выгрузить
    весь банк вопросов вместе с ответами через /api/questions/full,
    а также править и удалять вопросы.
    """
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return user


# Типы параметров обработчика: `user: CurrentUser` вместо
# `user: User = Depends(require_user)` в каждой сигнатуре
CurrentUser = Annotated[User, Depends(require_user)]
AdminUser = Annotated[User, Depends(require_admin)]
