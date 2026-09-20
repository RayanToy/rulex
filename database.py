import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from models import Base

# Каталог изменяемых данных. Отделён от data/ — там лежат частотные словари,
# которые версионируются вместе с кодом. В Docker сюда монтируется том,
# поэтому база переживает пересоздание контейнера.
VAR_DIR = Path(os.getenv("RULEX_DATA_DIR") or Path(__file__).parent / "var")
VAR_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = VAR_DIR / "questions.db"

DATABASE_URL = f"sqlite:///{DB_PATH}"
ASYNC_DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# Логирование SQL шумит в проде, поэтому по умолчанию выключено.
SQL_ECHO = (os.getenv("RULEX_SQL_ECHO") or "").strip().lower() in {"1", "true", "yes"}

# Синхронный движок нужен только для создания таблиц на старте.
engine = create_engine(DATABASE_URL, echo=SQL_ECHO)

async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=SQL_ECHO)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


def init_db():
    """Создание всех таблиц в базе данных"""
    Base.metadata.create_all(bind=engine)


async def get_db():
    """Получение сессии базы данных"""
    async with AsyncSessionLocal() as session:
        yield session
