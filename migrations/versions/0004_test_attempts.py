"""Попытки теста: какие вопросы выданы и кому.

Раньше /api/test/complete принимал ответы на любые существующие вопросы
и считал процент от присланного: можно было сдать «тест» из одних
знакомых вопросов, пропустить трудные или по одному вопросу выяснять,
какой вариант верный. Теперь старт теста создаёт попытку с набором
выданных вопросов, а проверка принимает ответы только на них и только
один раз.

user_id пуст у гостевых попыток — это тест без регистрации, его
результат не сохраняется. Хранится SHA-256 идентификатора попытки,
как и у сессий.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-25
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "test_attempts",
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("word_class", sa.Integer(), nullable=False),
        sa.Column("question_ids", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("token_hash"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"],
                                name="fk_test_attempts_user_id", ondelete="CASCADE"),
    )
    op.create_index("ix_test_attempts_user_id", "test_attempts", ["user_id"])
    # по истечению чистятся брошенные попытки
    op.create_index("ix_test_attempts_expires_at", "test_attempts", ["expires_at"])


def downgrade() -> None:
    op.drop_table("test_attempts")
