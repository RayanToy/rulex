"""Внешние ключи вместо «голых» Integer.

До этой ревизии user_id, question_id, test_result_id и created_by были
обычными числами без ограничений. Эндпоинт DELETE /api/questions/{id}
удалял вопрос, а ответы учеников продолжали ссылаться на несуществующую
запись.

Поведение при удалении выбрано для каждой связи отдельно:

  test_results.user_id        -> users           CASCADE
      результаты принадлежат ученику; удаление ученика удаляет и их
  test_answers.test_result_id -> test_results    CASCADE
      ответ — часть результата
  test_answers.question_id    -> questions       SET NULL
      удаление вопроса из банка НЕ должно стирать историю учеников;
      текст ответа и верный вариант хранятся в самой строке, так что
      история остаётся читаемой. Для этого колонка стала nullable
  questions.created_by        -> users           SET NULL
      уход преподавателя не должен уничтожать банк вопросов
  generation_logs.question_id -> questions       CASCADE
      журнал генерации принадлежит вопросу

Перед добавлением ограничений висячие ссылки вычищаются, иначе
ограничение объявлено, а данные его уже нарушают. Удаляются только
строки, которые по семантике связи удалились бы каскадом; там, где
связь SET NULL, ссылка обнуляется, а строка остаётся.

SQLite применяет внешние ключи, только если на соединении включено
PRAGMA foreign_keys=ON — это делает database.py. Без него ограничения
объявлены, но молча не работают.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-24
"""
import logging

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

log = logging.getLogger("alembic.runtime.migration")


def _run(sql: str) -> int:
    return op.get_bind().execute(sa.text(sql)).rowcount or 0


def upgrade() -> None:
    # 1. Висячие ссылки там, где связь каскадная: строки удаляются
    n = _run("DELETE FROM test_answers WHERE test_result_id IN "
             "(SELECT id FROM test_results WHERE user_id NOT IN (SELECT id FROM users))")
    n += _run("DELETE FROM test_results WHERE user_id NOT IN (SELECT id FROM users)")
    n += _run("DELETE FROM test_answers WHERE test_result_id NOT IN (SELECT id FROM test_results)")
    if n:
        log.info("удалено висячих результатов и ответов: %d", n)

    # 2. question_id становится nullable — иначе SET NULL невозможен
    with op.batch_alter_table("test_answers") as batch:
        batch.alter_column("question_id", existing_type=sa.Integer(), nullable=True)

    # 3. Висячие ссылки там, где связь SET NULL: строка остаётся
    n = _run("UPDATE test_answers SET question_id = NULL "
             "WHERE question_id IS NOT NULL AND question_id NOT IN (SELECT id FROM questions)")
    n += _run("UPDATE questions SET created_by = NULL "
              "WHERE created_by IS NOT NULL AND created_by NOT IN (SELECT id FROM users)")
    n += _run("UPDATE generation_logs SET question_id = NULL "
              "WHERE question_id IS NOT NULL AND question_id NOT IN (SELECT id FROM questions)")
    if n:
        log.info("обнулено висячих ссылок: %d", n)

    # 4. Сами ограничения. SQLite не умеет ALTER TABLE ADD CONSTRAINT,
    #    поэтому batch-режим пересоздаёт таблицу и копирует данные.
    with op.batch_alter_table("test_results") as batch:
        batch.create_foreign_key("fk_test_results_user_id", "users",
                                 ["user_id"], ["id"], ondelete="CASCADE")
    with op.batch_alter_table("test_answers") as batch:
        batch.create_foreign_key("fk_test_answers_test_result_id", "test_results",
                                 ["test_result_id"], ["id"], ondelete="CASCADE")
        batch.create_foreign_key("fk_test_answers_question_id", "questions",
                                 ["question_id"], ["id"], ondelete="SET NULL")
    with op.batch_alter_table("questions") as batch:
        batch.create_foreign_key("fk_questions_created_by", "users",
                                 ["created_by"], ["id"], ondelete="SET NULL")
    with op.batch_alter_table("generation_logs") as batch:
        batch.create_foreign_key("fk_generation_logs_question_id", "questions",
                                 ["question_id"], ["id"], ondelete="CASCADE")


def downgrade() -> None:
    # question_id остаётся nullable: после upgrade в нём могут быть NULL,
    # и вернуть NOT NULL без потери строк нельзя.
    with op.batch_alter_table("generation_logs") as batch:
        batch.drop_constraint("fk_generation_logs_question_id", type_="foreignkey")
    with op.batch_alter_table("questions") as batch:
        batch.drop_constraint("fk_questions_created_by", type_="foreignkey")
    with op.batch_alter_table("test_answers") as batch:
        batch.drop_constraint("fk_test_answers_question_id", type_="foreignkey")
        batch.drop_constraint("fk_test_answers_test_result_id", type_="foreignkey")
    with op.batch_alter_table("test_results") as batch:
        batch.drop_constraint("fk_test_results_user_id", type_="foreignkey")
