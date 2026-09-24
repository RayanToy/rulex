import hashlib
import secrets
from datetime import timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import delete, select

from database import AsyncSessionLocal
from models import Session, utcnow

# Argon2id — алгоритм, предназначенный для паролей: медленный и требовательный
# к памяти. Раньше здесь был одинарный SHA-256 с солью: он считается мгновенно,
# и перебор по словарю на GPU идёт миллиардами попыток в секунду.
_hasher = PasswordHasher()

SESSION_TTL = timedelta(days=7)


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


def _hash_token(token: str) -> str:
    """Токен в базе не хранится — только его SHA-256.

    Медленный хеш вроде Argon2 здесь не нужен: токен случайный,
    256 бит энтропии, перебирать нечего. Нужно лишь, чтобы утечка
    базы не давала готовых токенов для входа.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def create_session(user_id: int) -> str:
    """Создать сессию и вернуть токен для куки.

    Раньше сессии жили в словаре внутри процесса: перезапуск разлогинивал
    всех, а при нескольких воркерах вход не работал вовсе — токен,
    выданный одним процессом, другой не знал.
    """
    token = secrets.token_urlsafe(32)
    now = utcnow()
    async with AsyncSessionLocal() as db:
        # Попутно убираем истёкшие сессии этого пользователя, чтобы таблица не росла
        await db.execute(delete(Session).where(Session.user_id == user_id,
                                               Session.expires_at < now))
        db.add(Session(token_hash=_hash_token(token), user_id=user_id,
                       created_at=now, expires_at=now + SESSION_TTL))
        await db.commit()
    return token


async def get_user_id_from_token(token: str | None) -> int | None:
    """Кому принадлежит токен; None — если сессии нет или она истекла."""
    if not token:
        return None
    async with AsyncSessionLocal() as db:
        record = (await db.execute(
            select(Session).where(Session.token_hash == _hash_token(token))
        )).scalar_one_or_none()
        if record is None:
            return None
        if utcnow() > record.expires_at:
            await db.delete(record)
            await db.commit()
            return None
        return record.user_id


async def delete_session(token: str) -> None:
    """Выход: сессия удаляется из базы, токен перестаёт работать сразу."""
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Session).where(Session.token_hash == _hash_token(token)))
        await db.commit()
