import hashlib
import secrets
from datetime import timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from models import utcnow

# Argon2id — алгоритм, предназначенный для паролей: медленный и требовательный
# к памяти. Раньше здесь был одинарный SHA-256 с солью: он считается мгновенно,
# и перебор по словарю на GPU идёт миллиардами попыток в секунду.
_hasher = PasswordHasher()

sessions = {}


def hash_password(password: str) -> str:
    """Хеширование пароля через Argon2id"""
    return _hasher.hash(password)


def _verify_legacy(plain_password: str, hashed_password: str) -> bool:
    """Проверка старого формата 'соль$sha256' — для входа тех, кто
    зарегистрировался до перехода на Argon2. Сравнение постоянное по времени."""
    try:
        salt, pwd_hash = hashed_password.split('$', 1)
    except ValueError:
        return False
    check = hashlib.sha256((plain_password + salt).encode()).hexdigest()
    return secrets.compare_digest(check, pwd_hash)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверка пароля. Понимает и Argon2, и старый формат."""
    if not hashed_password:
        return False
    if hashed_password.startswith('$argon2'):
        try:
            return _hasher.verify(hashed_password, plain_password)
        except (VerifyMismatchError, VerificationError, InvalidHashError, ValueError):
            # ValueError покрывает и UnicodeEncodeError: на повреждённой
            # записи в БД argon2 бросает именно его, и вход отвечал бы 500.
            return False
    return _verify_legacy(plain_password, hashed_password)


def needs_rehash(hashed_password: str) -> bool:
    """Нужно ли пересохранить хеш: старый формат или устаревшие параметры Argon2.

    Позволяет перевести пользователей на Argon2 незаметно — при первом
    успешном входе, не требуя смены пароля.
    """
    if not hashed_password or not hashed_password.startswith('$argon2'):
        return True
    try:
        return _hasher.check_needs_rehash(hashed_password)
    except InvalidHashError:
        return True


def create_session(user_id: int) -> str:
    """Создание сессии"""
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "user_id": user_id,
        "created_at": utcnow(),
        "expires_at": utcnow() + timedelta(days=7)
    }
    return token


def get_session(token: str) -> dict | None:
    """Получение сессии"""
    if token not in sessions:
        return None
    session = sessions[token]
    if utcnow() > session["expires_at"]:
        del sessions[token]
        return None
    return session


def delete_session(token: str):
    """Удаление сессии"""
    if token in sessions:
        del sessions[token]


def get_user_id_from_token(token: str) -> int | None:
    """Получение user_id из токена"""
    session = get_session(token)
    if session:
        return session["user_id"]
    return None
