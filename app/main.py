"""Сборка приложения: жизненный цикл, статика, роутеры.

Запуск:
    uvicorn app.main:app --reload
"""
import os
import sys

if sys.platform == "win32":
    # reconfigure есть не у всякого stdout: под тестами и некоторыми
    # серверами поток бывает обёрнут и такого метода не имеет.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

import asyncio  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app import PROJECT_ROOT  # noqa: E402
from app.api import assessment, auth, pages, questions  # noqa: E402
from app.core.database import migrate  # noqa: E402
from app.services.wordlists import get_word_manager  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Миграции Alembic вместо create_all: тот не умеет менять уже
    # существующие таблицы. Синхронные — поэтому в потоке.
    await asyncio.to_thread(migrate)
    # Словари прогреваются при старте, а не на первом запросе: чтение
    # ~12.5 МБ CSV занимает около 0.9 с, и в обработчике это блокировало
    # event loop, подвешивая сервер для всех остальных клиентов.
    await asyncio.to_thread(get_word_manager)
    yield


app = FastAPI(title="RuLex", lifespan=lifespan)

# Путь от корня проекта, а не от текущего каталога: раньше приложение
# находило статику, только если его запускали из корня репозитория.
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

app.include_router(pages.router)
app.include_router(auth.router)
app.include_router(assessment.router)
app.include_router(questions.router)
