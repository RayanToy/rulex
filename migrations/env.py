"""Окружение Alembic.

Адрес базы берётся из database.DATABASE_URL, а не из alembic.ini: у
приложения и миграций должен быть один источник правды, иначе миграции
легко накатить не на ту базу.

Миграции идут через собственный движок, БЕЗ включения внешних ключей.
SQLite не умеет менять ограничения таблицы через ALTER TABLE, поэтому
Alembic в batch-режиме пересоздаёт таблицу и копирует в неё данные;
с включёнными внешними ключами это копирование спотыкалось бы о них.
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models  # noqa: E402

config = context.config
target_metadata = models.Base.metadata

# При вызове из приложения логирование уже настроено — не перетираем его
if config.config_file_name is not None and not config.attributes.get("skip_logging"):
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def _url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from database import DATABASE_URL
    return DATABASE_URL


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata,
                      literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite: изменения ограничений — через пересоздание таблицы
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
