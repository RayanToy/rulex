from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from models import Base

DATABASE_URL = "sqlite:///./questions.db"
ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./questions.db"

# Синхронный движок для создания таблиц
engine = create_engine(DATABASE_URL, echo=True)

# Асинхронный движок для работы с FastAPI
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=True)

# Фабрика сессий
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

def init_db():
    """Создание всех таблиц в базе данных"""
    Base.metadata.create_all(bind=engine)

async def get_db():
    """Получение сессии базы данных"""
    async with AsyncSessionLocal() as session:
        yield session