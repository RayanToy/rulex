"""Целостность теста и контроль доступа.

Эти проверки закрывают два дефекта, из-за которых платформа оценки
знаний не давала достоверных результатов:

1. Результат считался в браузере и присылался на сервер готовым —
   его можно было подделать из DevTools за десять секунд.
2. Поле is_admin существовало, но не проверялось: любой ученик мог
   выгрузить банк вопросов вместе с правильными ответами.
"""
import pytest


ANSWER_FIELDS = ("correct", "correct_answer", "target_word")


def start_test(client, cookies, grade=6):
    client.cookies.clear()
    client.cookies.update(cookies)
    response = client.post("/api/test/start", json={"grade": grade})
    assert response.status_code == 200, response.text
    return response.json()["questions"]


class TestAnswerNotLeaked:
    def test_start_payload_has_no_answer(self, client, pupil_cookies, question_bank):
        for question in start_test(client, pupil_cookies):
            leaked = [f for f in ANSWER_FIELDS if f in question]
            assert not leaked, f"клиенту ушли поля с ответом: {leaked}"

    def test_public_payload_has_no_answer(self, client, question_bank):
        client.cookies.clear()
        response = client.post("/api/public/test/auto-start", json={"grade": 6})
        assert response.status_code == 200, response.text
        for question in response.json()["questions"]:
            leaked = [f for f in ANSWER_FIELDS if f in question]
            assert not leaked, f"клиенту ушли поля с ответом: {leaked}"


class TestServerSideScoring:
    def test_wrong_answers_score_zero(self, client, pupil_cookies, question_bank):
        questions = start_test(client, pupil_cookies)
        payload = [{"question_id": q["id"], "answer": "заведомо-неверно"} for q in questions]
        result = client.post("/api/test/complete",
                             json={"answers": payload, "grade": 6}).json()
        assert result["score"] == 0

    def test_correct_answers_score_full(self, client, pupil_cookies, question_bank):
        questions = start_test(client, pupil_cookies)
        payload = [{"question_id": q["id"], "answer": question_bank[q["id"]]}
                   for q in questions]
        result = client.post("/api/test/complete",
                             json={"answers": payload, "grade": 6}).json()
        assert result["score"] == result["total"] == len(questions)
        assert result["percentage"] == 100.0

    def test_client_supplied_verdict_is_ignored(self, client, pupil_cookies, question_bank):
        """Старый формат с is_correct=True не должен давать очков."""
        questions = start_test(client, pupil_cookies)
        payload = [{"question_id": q["id"], "answer": "мимо", "is_correct": True}
                   for q in questions]
        result = client.post("/api/test/complete",
                             json={"answers": payload, "grade": 6}).json()
        assert result["score"] == 0

    def test_unknown_question_rejected(self, client, pupil_cookies, question_bank):
        response = client.post("/api/test/complete", json={
            "answers": [{"question_id": 10**9, "answer": "что-нибудь"}], "grade": 6})
        assert response.status_code == 400


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
