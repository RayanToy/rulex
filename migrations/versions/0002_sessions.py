"""Сессии в базе вместо словаря в памяти процесса.

Раньше сессии жили в словаре auth.sessions: любой перезапуск разлогинивал
всех, а при нескольких воркерах вход не работал вовсе — токен, выданный
одним процессом, другой не знал.

Хранится не токен, а его SHA-256: утечка базы не даёт готовых токенов.
Медленный хеш вроде Argon2 здесь не нужен — токен случайный, 256 бит
энтропии, перебирать нечего.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-24
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("token_hash"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"],
                                name="fk_sessions_user_id", ondelete="CASCADE"),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    # по истечению чистятся устаревшие записи
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])


def downgrade() -> None:
    op.drop_table("sessions")
