"""Общие фикстуры.

Переменные окружения выставляются ДО импорта приложения: app/core/database.py
определяет путь к БД на этапе импорта модуля.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["RULEX_DATA_DIR"] = tempfile.mkdtemp(prefix="rulex-tests-")
# Банк вопросов тесты наполняют сами; готовый набор для демо не грузится.
os.environ["RULEX_SEED_BANK"] = ""
# Генератор в этих тестах не вызывается, но конструктор требует ключ.
os.environ.setdefault("ANTHROPIC_API_KEY", "tests-placeholder")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def admin_cookies(client):
    """Первый зарегистрированный пользователь получает права администратора."""
    client.cookies.clear()
    response = client.post("/api/auth/register", json={
        "username": "teacher", "email": "teacher@example.org",
        "password": "pw-teacher-123", "grade": 6,
    })
    assert response.status_code == 200, response.text
    assert response.json()["user"]["is_admin"] is True
    return dict(client.cookies)


@pytest.fixture(scope="session")
def pupil_cookies(client, admin_cookies):
    client.cookies.clear()
    response = client.post("/api/auth/register", json={
        "username": "pupil", "email": "pupil@example.org",
        "password": "pw-pupil-123", "grade": 6,
    })
    assert response.status_code == 200, response.text
    assert response.json()["user"]["is_admin"] is False
    return dict(client.cookies)


@pytest.fixture(scope="session")
def question_bank(client, admin_cookies):
    """Несколько вопросов: /api/test/start требует минимум пять."""
    client.cookies.clear()
    client.cookies.update(admin_cookies)
    words = [
        ("кот", "Домашнее животное, ловит мышей", ["стул", "окно", "ветер"]),
        ("река", "Постоянный водный поток", ["гора", "лампа", "книга"]),
        ("хлеб", "Продукт из муки, выпекается в печи", ["камень", "стекло", "туча"]),
        ("город", "Крупный населённый пункт", ["ложка", "перо", "нитка"]),
        ("лампа", "Прибор для освещения помещения", ["берег", "пирог", "туман"]),
        ("мороз", "Сильный холод зимой", ["песок", "сахар", "провод"]),
    ]
    created = {}
    for word, definition, distractors in words:
        response = client.post("/api/questions", json={
            "target_word": word, "definition": definition, "correct_answer": word,
            "distractor_1": distractors[0], "distractor_2": distractors[1],
            "distractor_3": distractors[2],
            "word_class": 6, "frequency_type": "medium", "difficulty": 5,
        })
        assert response.status_code == 200, response.text
        created[response.json()["id"]] = word
    return created
