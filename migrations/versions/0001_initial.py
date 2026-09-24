"""Исходная схема — ровно та, что создавал create_all до появления миграций.

Ревизия воспроизводит существующую схему байт в байт (5 таблиц, 10 индексов
с теми же именами), поэтому уже развёрнутую базу можно пометить ею
(`alembic stamp 0001`) и накатить поверх остальные ревизии без потери
данных. database.migrate() делает это сам, если видит таблицы без
alembic_version.

Revision ID: 0001
Revises:
Create Date: 2026-09-24
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("email", sa.String(100), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(100), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("is_admin", sa.Boolean(), nullable=True),
        sa.Column("grade", sa.Integer(), nullable=True),
        sa.Column("current_level", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("last_login", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "questions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("target_word", sa.String(100), nullable=False),
        sa.Column("definition", sa.Text(), nullable=False),
        sa.Column("correct_answer", sa.String(100), nullable=False),
        sa.Column("distractor_1", sa.String(100), nullable=True),
        sa.Column("distractor_2", sa.String(100), nullable=True),
        sa.Column("distractor_3", sa.String(100), nullable=True),
        sa.Column("word_class", sa.Integer(), nullable=True),
        sa.Column("frequency_type", sa.String(20), nullable=True),
        sa.Column("difficulty", sa.Integer(), nullable=True),
        sa.Column("part_of_speech", sa.String(20), nullable=True),
        sa.Column("is_approved", sa.Boolean(), nullable=True),
        sa.Column("generation_log", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_questions_id", "questions", ["id"])
    op.create_index("ix_questions_target_word", "questions", ["target_word"])

    op.create_table(
        "test_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("total_questions", sa.Integer(), nullable=False),
        sa.Column("percentage", sa.Float(), nullable=False),
        sa.Column("high_freq_correct", sa.Integer(), nullable=True),
        sa.Column("high_freq_total", sa.Integer(), nullable=True),
        sa.Column("medium_freq_correct", sa.Integer(), nullable=True),
        sa.Column("medium_freq_total", sa.Integer(), nullable=True),
        sa.Column("low_freq_correct", sa.Integer(), nullable=True),
        sa.Column("low_freq_total", sa.Integer(), nullable=True),
        sa.Column("grade_tested", sa.Integer(), nullable=False),
        sa.Column("level_achieved", sa.String(20), nullable=False),
        sa.Column("max_difficulty_reached", sa.Integer(), nullable=True),
        sa.Column("recommendation", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_test_results_id", "test_results", ["id"])
    op.create_index("ix_test_results_user_id", "test_results", ["user_id"])

    op.create_table(
        "test_answers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("test_result_id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("user_answer", sa.String(100), nullable=True),
        sa.Column("correct_answer", sa.String(100), nullable=True),
        sa.Column("difficulty_at_answer", sa.Integer(), nullable=True),
        sa.Column("answered_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_test_answers_id", "test_answers", ["id"])
    op.create_index("ix_test_answers_test_result_id", "test_answers", ["test_result_id"])

    op.create_table(
        "generation_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=True),
        sa.Column("step", sa.String(50), nullable=False),
        sa.Column("input_data", sa.Text(), nullable=True),
        sa.Column("output_data", sa.Text(), nullable=True),
        sa.Column("llm_prompt", sa.Text(), nullable=True),
        sa.Column("llm_response", sa.Text(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_generation_logs_id", "generation_logs", ["id"])


def downgrade() -> None:
    for table in ("generation_logs", "test_answers", "test_results", "questions", "users"):
        op.drop_table(table)
