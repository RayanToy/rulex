import os
from pathlib import Path

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent

# Каталог изменяемых данных. Отделён от data/ — там лежат частотные словари,
# которые версионируются вместе с кодом. В Docker сюда монтируется том,
# поэтому база переживает пересоздание контейнера.
VAR_DIR = Path(os.getenv("RULEX_DATA_DIR") or ROOT / "var")
VAR_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = VAR_DIR / "questions.db"

DATABASE_URL = f"sqlite:///{DB_PATH}"
ASYNC_DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# Логирование SQL шумит в проде, поэтому по умолчанию выключено.
SQL_ECHO = (os.getenv("RULEX_SQL_ECHO") or "").strip().lower() in {"1", "true", "yes"}

# Синхронный движок — для миграций и служебных скриптов.
engine = create_engine(DATABASE_URL, echo=SQL_ECHO)

async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=SQL_ECHO)


def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
    """SQLite применяет внешние ключи, только если это включено на КАЖДОМ
    соединении. Без этой строки ограничения в схеме объявлены, но молча
    не работают: удаление ученика не удалит его сессии, а ответ можно
    записать к несуществующему результату."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


event.listen(engine, "connect", _enable_foreign_keys)
event.listen(async_engine.sync_engine, "connect", _enable_foreign_keys)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


def migrate(url: str | None = None) -> None:
    """Привести схему базы к последней ревизии Alembic.

    Раньше таблицы создавал create_all: он умеет только создать таблицу
    с нуля и не может изменить уже существующую — добавить внешний ключ
    или новую колонку в развёрнутую базу им нельзя.

    База, созданная create_all до появления миграций, распознаётся по
    таблицам без alembic_version и помечается исходной ревизией 0001 — её
    схема повторяет прежний create_all байт в байт. Дальше накатываются
    остальные ревизии, данные сохраняются.
    """
    from alembic import command
    from alembic.config import Config

    url = url or DATABASE_URL
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["skip_logging"] = True
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    # configparser трактует % как подстановку — экранируем
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))

    probe = engine if url == DATABASE_URL else create_engine(url)
    try:
        tables = set(inspect(probe).get_table_names())
    finally:
        if probe is not engine:
            probe.dispose()
    if "users" in tables and "alembic_version" not in tables:
        command.stamp(cfg, "0001")
    command.upgrade(cfg, "head")


async def get_db():
    """Получение сессии базы данных"""
    async with AsyncSessionLocal() as session:
        yield session
