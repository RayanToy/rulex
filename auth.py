import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

sessions = {}


def hash_password(password: str) -> str:
    """Хеширование пароля через SHA256"""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${pwd_hash}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверка пароля"""
    try:
        salt, pwd_hash = hashed_password.split('$')
        check_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
        return check_hash == pwd_hash
    except:
        return False


def create_session(user_id: int) -> str:
    """Создание сессии"""
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(days=7)
    }
    return token


def get_session(token: str) -> Optional[dict]:
    """Получение сессии"""
    if token not in sessions:
        return None
    session = sessions[token]
    if datetime.utcnow() > session["expires_at"]:
        del sessions[token]
        return None
    return session


def delete_session(token: str):
    """Удаление сессии"""
    if token in sessions:
        del sessions[token]


def get_user_id_from_token(token: str) -> Optional[int]:
    """Получение user_id из токена"""
    session = get_session(token)
    if session:
        return session["user_id"]
    return None