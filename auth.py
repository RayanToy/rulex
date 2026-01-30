from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import secrets

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
sessions = {}


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(days=7)
    }
    return token


def get_session(token: str) -> Optional[dict]:
    if token not in sessions:
        return None
    session = sessions[token]
    if datetime.utcnow() > session["expires_at"]:
        del sessions[token]
        return None
    return session


def delete_session(token: str):
    if token in sessions:
        del sessions[token]


def get_user_id_from_token(token: str) -> Optional[int]:
    session = get_session(token)
    if session:
        return session["user_id"]
    return None