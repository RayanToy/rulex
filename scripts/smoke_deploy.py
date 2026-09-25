#!/usr/bin/env python
"""Проверка развёрнутого экземпляра: страница, банк вопросов, гостевой тест.

    python scripts/smoke_deploy.py https://rulex.example.ru
    python scripts/smoke_deploy.py http://localhost:8000 --register   # + регистрация
    python scripts/smoke_deploy.py http://localhost:8000 --login      # + вход тем же пользователем

По умолчанию только чтение и гостевой тест — на публичном демо ничего
не создаётся. --register и --login нужны CI: после перезапуска контейнера
вход проверяет, что база пережила рестарт. Только стандартная библиотека:
скрипт запускается и там, где зависимостей проекта нет.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BANK = Path(__file__).resolve().parent.parent / "data" / "demo_questions.json"
USER = {"username": "smoke-user", "email": "smoke@example.org", "password": "smoke-pass-123", "grade": 6}


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def call(self, method: str, path: str, payload: dict | None = None) -> tuple[int, object]:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as exc:
            body, status = exc.read().decode("utf-8"), exc.code
        try:
            return status, json.loads(body)
        except json.JSONDecodeError:
            return status, body


def check(condition: bool, message: str) -> None:
    print(("OK   " if condition else "FAIL ") + message)
    if not condition:
        raise SystemExit(1)


def wait_ready(client: Client, seconds: int = 90) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if client.call("GET", "/")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(2)
    check(False, f"сервер не ответил за {seconds} с")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--login", action="store_true")
    args = parser.parse_args()

    client = Client(args.url)
    wait_ready(client)

    status, page = client.call("GET", "/")
    check(status == 200 and "RuLex" in page, "главная страница открывается")

    status, grades = client.call("GET", "/api/public/grades")
    ready = [g for g in grades if g["ready"]] if status == 200 else []
    check(bool(ready), f"в банке есть готовые классы: {[g['word_class'] for g in ready]}")
    if BANK.exists():
        expected = len(json.loads(BANK.read_text(encoding="utf-8"))["questions"])
        total = sum(g["questions"] for g in grades)
        check(total == expected, f"банк загружен без дублей: {total} из {expected}")

    status, test = client.call("POST", "/api/public/test/auto-start", {"grade": ready[0]["word_class"]})
    check(status == 200 and test["questions"], f"гостевой тест выдан: {len(test.get('questions', []))} вопросов")
    check(all("correct_answer" not in q for q in test["questions"]), "ответы не уходят клиенту")
    answers = [{"question_id": q["id"], "answer": q["options"][0]} for q in test["questions"]]
    status, result = client.call("POST", "/api/public/test/complete",
                                 {"attempt_id": test["attempt_id"], "answers": answers})
    check(status == 200 and result["total"] == len(answers), f"гостевой тест проверен: {result.get('score')}/{result.get('total')}")
    status, _ = client.call("POST", "/api/public/test/complete",
                            {"attempt_id": test["attempt_id"], "answers": answers})
    check(status == 400, "попытка одноразовая")

    if args.register:
        status, body = client.call("POST", "/api/auth/register", USER)
        check(status == 200, "регистрация")
        check(body["user"]["is_admin"] is False, "первый зарегистрированный не становится администратором")
    if args.login:
        status, _ = client.call("POST", "/api/auth/login",
                                {"username": USER["username"], "password": USER["password"]})
        check(status == 200, "вход после перезапуска — база сохранилась")

    print("Всё в порядке")
    return 0


if __name__ == "__main__":
    sys.exit(main())
