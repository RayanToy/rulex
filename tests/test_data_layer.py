"""Миграции, сессии в базе и внешние ключи.

Миграции проверяются на отдельных временных базах. «Старая» база
эмулируется точно: ревизия 0001 повторяет схему прежнего create_all
байт в байт, поэтому достаточно накатить её и удалить alembic_version —
ровно в таком состоянии оставлял базу create_all.
"""
import hashlib
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import database, security  # noqa: E402
from app.core.models import utcnow  # noqa: E402


def _cfg(url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["skip_logging"] = True
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _fks(conn, table):
    """(колонка, таблица-цель, ON DELETE) для каждого внешнего ключа."""
    return {(r[3], r[2], r[6]) for r in conn.execute(text(f"PRAGMA foreign_key_list({table})"))}


class TestMigrations:
    def test_fresh_database_reaches_head_with_foreign_keys(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'fresh.db'}"
        database.migrate(url)
        eng = create_engine(url)
        with eng.connect() as conn:
            assert "sessions" in inspect(conn).get_table_names()
            assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0003"
            assert ("question_id", "questions", "SET NULL") in _fks(conn, "test_answers")
            assert ("test_result_id", "test_results", "CASCADE") in _fks(conn, "test_answers")
            assert ("user_id", "users", "CASCADE") in _fks(conn, "sessions")
        eng.dispose()

    def test_legacy_database_is_upgraded_without_data_loss(self, tmp_path):
        """База из времён create_all: таблицы есть, alembic_version нет,
        а после удаления вопроса остался ответ с висячей ссылкой."""
        url = f"sqlite:///{tmp_path / 'legacy.db'}"
        command.upgrade(_cfg(url), "0001")
        eng = create_engine(url)
        with eng.begin() as conn:
            conn.execute(text("DROP TABLE alembic_version"))
            conn.execute(text("INSERT INTO users (id, username, email, hashed_password) "
                              "VALUES (1, 'teacher', 't@x', 'h')"))
            conn.execute(text("INSERT INTO test_results (id, user_id, score, total_questions, "
                              "percentage, grade_tested, level_achieved) "
                              "VALUES (1, 1, 3, 5, 60.0, 6, 'low')"))
            # question_id=999 — вопрос удалён через DELETE /api/questions/{id}
            conn.execute(text("INSERT INTO test_answers (id, test_result_id, question_id, "
                              "is_correct, user_answer, correct_answer) "
                              "VALUES (1, 1, 999, 1, 'кот', 'кот')"))

        database.migrate(url)

        with eng.connect() as conn:
            assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0003"
            assert conn.execute(text("SELECT username FROM users")).scalar() == "teacher"
            answer = conn.execute(text("SELECT question_id, user_answer FROM test_answers")).one()
            # висячая ссылка обнулена, а история ответа сохранена
            assert answer.question_id is None
            assert answer.user_answer == "кот"
        eng.dispose()

    def test_migrate_is_idempotent(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'twice.db'}"
        database.migrate(url)
        database.migrate(url)  # не должно падать и ничего не менять
        eng = create_engine(url)
        with eng.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM alembic_version")).scalar() == 1
        eng.dispose()


def _register(client, name):
    client.cookies.clear()
    response = client.post("/api/auth/register", json={
        "username": name, "email": f"{name}@example.org", "password": "pw-123456"})
    assert response.status_code == 200, response.text
    return response.cookies.get("session_token"), response.json()["user"]["id"]


# Администратор должен быть зарегистрирован ПЕРВЫМ: по правилу «первый
# пользователь — администратор» иначе админом стал бы пользователь из этих
# тестов, и фикстура admin_cookies падала бы. Без явной зависимости набор
# проходил или падал в зависимости от порядка запуска файлов.
@pytest.mark.usefixtures("admin_cookies")
class TestSessionsInDatabase:
    def test_no_in_memory_session_store(self):
        """Прежний словарь в памяти процесса терялся при каждом перезапуске."""
        assert not hasattr(security, "sessions")

    def test_session_is_persisted_and_token_is_hashed(self, client):
        token, user_id = _register(client, "session_owner")
        with database.engine.connect() as conn:
            stored = conn.execute(text("SELECT token_hash FROM sessions WHERE user_id = :u"),
                                  {"u": user_id}).scalars().all()
        assert hashlib.sha256(token.encode()).hexdigest() in stored
        assert token not in stored  # в базе нет готового токена

    def test_expired_session_is_rejected_and_removed(self, client):
        token, user_id = _register(client, "expired_user")
        # Строкой в формате SQLAlchemy для SQLite: голый datetime в text()
        # идёт через адаптер sqlite3, объявленный устаревшим в Python 3.12
        expired = str(utcnow() - timedelta(minutes=1))
        with database.engine.begin() as conn:
            conn.execute(text("UPDATE sessions SET expires_at = :t WHERE user_id = :u"),
                         {"t": expired, "u": user_id})
        client.cookies.clear()
        client.cookies.set("session_token", token)
        assert client.get("/api/auth/me").status_code == 401
        with database.engine.connect() as conn:
            left = conn.execute(text("SELECT count(*) FROM sessions WHERE user_id = :u"),
                                {"u": user_id}).scalar()
        assert left == 0

    def test_logout_invalidates_token(self, client):
        token, _ = _register(client, "logout_user")
        client.cookies.clear()
        client.cookies.set("session_token", token)
        assert client.get("/api/auth/me").status_code == 200
        client.post("/api/auth/logout")
        client.cookies.clear()
        client.cookies.set("session_token", token)
        assert client.get("/api/auth/me").status_code == 401


@pytest.mark.usefixtures("admin_cookies")
class TestForeignKeysAreEnforced:
    """Ограничение в схеме ничего не значит, если SQLite его не применяет:
    для этого на каждом соединении нужен PRAGMA foreign_keys=ON."""

    def test_pragma_is_on_for_app_connections(self, client):
        with database.engine.connect() as conn:
            assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1

    def test_answer_to_missing_result_is_rejected(self, client):
        with pytest.raises(IntegrityError), database.engine.begin() as conn:
            conn.execute(text("INSERT INTO test_answers (test_result_id, question_id, is_correct) "
                              "VALUES (999999, NULL, 0)"))

    def test_deleting_user_cascades_to_sessions(self, client):
        _, user_id = _register(client, "to_be_deleted")
        with database.engine.begin() as conn:
            conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})
        with database.engine.connect() as conn:
            left = conn.execute(text("SELECT count(*) FROM sessions WHERE user_id = :u"),
                                {"u": user_id}).scalar()
        assert left == 0
