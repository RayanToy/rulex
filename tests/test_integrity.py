"""Целостность теста и контроль доступа.

Эти проверки закрывают дефекты, из-за которых платформа оценки знаний
не давала достоверных результатов:

1. Результат считался в браузере и присылался на сервер готовым —
   его можно было подделать из DevTools за десять секунд.
2. Поле is_admin существовало, но не проверялось: любой ученик мог
   выгрузить банк вопросов вместе с правильными ответами.
3. Проверка принимала ответы на любые вопросы и считала процент от
   присланного: можно было сдать «тест» из знакомых вопросов, пропустить
   трудные или по одному вопросу выяснять верный вариант. Теперь тест
   привязан к попытке с набором выданных вопросов.
"""
import secrets
from datetime import timedelta

import pytest
from sqlalchemy import text

from app.core import database
from app.core.models import utcnow

ANSWER_FIELDS = ("correct", "correct_answer", "target_word")


def start_test(client, cookies=None, grade=6, path="/api/test/start"):
    client.cookies.clear()
    if cookies:
        client.cookies.update(cookies)
    response = client.post(path, json={"grade": grade})
    assert response.status_code == 200, response.text
    return response.json()


def start_guest_test(client, grade=6):
    return start_test(client, None, grade, path="/api/public/test/auto-start")


def complete(client, test, answers, path="/api/test/complete"):
    return client.post(path, json={"attempt_id": test["attempt_id"], "answers": answers})


def right_answers(test, bank):
    return [{"question_id": q["id"], "answer": bank[q["id"]]} for q in test["questions"]]


def results_count():
    with database.engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM test_results")).scalar()


class TestAnswerNotLeaked:
    def test_start_payload_has_no_answer(self, client, pupil_cookies, question_bank):
        for question in start_test(client, pupil_cookies)["questions"]:
            leaked = [f for f in ANSWER_FIELDS if f in question]
            assert not leaked, f"клиенту ушли поля с ответом: {leaked}"

    def test_public_payload_has_no_answer(self, client, question_bank):
        for question in start_guest_test(client)["questions"]:
            leaked = [f for f in ANSWER_FIELDS if f in question]
            assert not leaked, f"клиенту ушли поля с ответом: {leaked}"


class TestServerSideScoring:
    def test_wrong_answers_score_zero(self, client, pupil_cookies, question_bank):
        test = start_test(client, pupil_cookies)
        payload = [{"question_id": q["id"], "answer": "заведомо-неверно"} for q in test["questions"]]
        assert complete(client, test, payload).json()["score"] == 0

    def test_correct_answers_score_full(self, client, pupil_cookies, question_bank):
        test = start_test(client, pupil_cookies)
        result = complete(client, test, right_answers(test, question_bank)).json()
        assert result["score"] == result["total"] == len(test["questions"])
        assert result["percentage"] == 100.0

    def test_client_supplied_verdict_is_ignored(self, client, pupil_cookies, question_bank):
        """Старый формат с is_correct=True не должен давать очков."""
        test = start_test(client, pupil_cookies)
        payload = [{"question_id": q["id"], "answer": "мимо", "is_correct": True}
                   for q in test["questions"]]
        assert complete(client, test, payload).json()["score"] == 0

    def test_grade_comes_from_attempt_not_request(self, client, pupil_cookies, question_bank):
        test = start_test(client, pupil_cookies, grade=6)
        response = client.post("/api/test/complete", json={
            "attempt_id": test["attempt_id"], "answers": right_answers(test, question_bank),
            "grade": 11})
        assert response.json()["grade"] == 6

    def test_recommendation_names_pupils_grade(self, client, pupil_cookies, question_bank):
        """Списки слов нумеруются на класс младше ученика: список 6 класса — это 7 класс."""
        test = start_test(client, pupil_cookies, grade=6)
        payload = [{"question_id": q["id"], "answer": "мимо"} for q in test["questions"]]
        assert "7 класса" in complete(client, test, payload).json()["recommendation"]


class TestAttempts:
    def test_attempt_is_single_use(self, client, pupil_cookies, question_bank):
        test = start_test(client, pupil_cookies)
        answers = right_answers(test, question_bank)
        assert complete(client, test, answers).status_code == 200
        assert complete(client, test, answers).status_code == 400

    def test_skipped_questions_count_as_wrong(self, client, pupil_cookies, question_bank):
        """Раньше процент считался от присланных ответов: пропустив трудные
        вопросы, можно было получить 100%."""
        test = start_test(client, pupil_cookies)
        result = complete(client, test, right_answers(test, question_bank)[:1]).json()
        assert result["total"] == len(test["questions"])
        assert result["score"] == 1

    def test_foreign_question_is_rejected_and_attempt_survives(self, client, pupil_cookies, question_bank):
        test = start_test(client, pupil_cookies)
        response = complete(client, test, [{"question_id": 10**9, "answer": "что-нибудь"}])
        assert response.status_code == 400
        # Отклонённая отправка попытку не тратит
        assert complete(client, test, right_answers(test, question_bank)).status_code == 200

    def test_unknown_attempt_is_rejected(self, client, pupil_cookies, question_bank):
        client.cookies.clear()
        client.cookies.update(pupil_cookies)
        response = client.post("/api/test/complete", json={
            "attempt_id": secrets.token_urlsafe(24), "answers": []})
        assert response.status_code == 400

    def test_someone_elses_attempt_is_rejected(self, client, pupil_cookies, admin_cookies, question_bank):
        test = start_test(client, pupil_cookies)
        client.cookies.clear()
        client.cookies.update(admin_cookies)
        assert complete(client, test, right_answers(test, question_bank)).status_code == 400

    def test_expired_attempt_is_rejected(self, client, pupil_cookies, question_bank):
        test = start_test(client, pupil_cookies)
        with database.engine.begin() as conn:
            conn.execute(text("UPDATE test_attempts SET expires_at = :t"),
                         {"t": str(utcnow() - timedelta(minutes=1))})
        assert complete(client, test, right_answers(test, question_bank)).status_code == 400


class TestGuestMode:
    """Тест без регистрации: проверяется на сервере, не сохраняется."""

    def test_guest_test_is_graded_and_not_saved(self, client, question_bank):
        test = start_guest_test(client)
        before = results_count()
        response = complete(client, test, right_answers(test, question_bank),
                            path="/api/public/test/complete")
        assert response.status_code == 200, response.text
        assert response.json()["percentage"] == 100.0
        assert results_count() == before

    def test_guest_attempt_is_not_accepted_as_pupils(self, client, pupil_cookies, question_bank):
        test = start_guest_test(client)
        client.cookies.update(pupil_cookies)
        assert complete(client, test, right_answers(test, question_bank)).status_code == 400

    def test_pupils_attempt_is_not_accepted_as_guests(self, client, pupil_cookies, question_bank):
        test = start_test(client, pupil_cookies)
        client.cookies.clear()
        response = complete(client, test, right_answers(test, question_bank),
                            path="/api/public/test/complete")
        assert response.status_code == 400

    def test_public_grades_lists_ready_classes(self, client, question_bank):
        client.cookies.clear()
        grades = {g["word_class"]: g for g in client.get("/api/public/grades").json()}
        assert grades[6]["ready"] is True
        assert grades[6]["questions"] >= len(question_bank)


class TestAccessControl:
    @pytest.mark.parametrize("method,path", [
        ("get", "/api/questions"),
        ("get", "/api/questions/full"),
        ("get", "/api/questions/random"),
        ("delete", "/api/questions/1"),
    ])
    def test_pupil_is_forbidden(self, client, pupil_cookies, question_bank, method, path):
        client.cookies.clear()
        client.cookies.update(pupil_cookies)
        assert getattr(client, method)(path).status_code == 403

    def test_admin_is_allowed(self, client, admin_cookies, question_bank):
        client.cookies.clear()
        client.cookies.update(admin_cookies)
        assert client.get("/api/questions/full").status_code == 200

    def test_anonymous_is_unauthorized(self, client, question_bank):
        client.cookies.clear()
        assert client.get("/api/questions/full").status_code == 401


class TestPublicEndpointDoesNotGenerate:
    def test_missing_grade_returns_503_instead_of_calling_llm(self, client, question_bank):
        """Открытый эндпоинт не должен запускать платную генерацию."""
        client.cookies.clear()
        response = client.post("/api/public/test/auto-start", json={"grade": 9})
        assert response.status_code == 503


class TestSessionCookie:
    def test_cookie_flags(self, client, question_bank):
        client.cookies.clear()
        response = client.post("/api/auth/login",
                               json={"username": "pupil", "password": "pw-pupil-123"})
        assert response.status_code == 200
        cookie = response.headers.get("set-cookie", "").lower()
        assert "httponly" in cookie
        assert "samesite=lax" in cookie
