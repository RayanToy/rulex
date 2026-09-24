"""Логика тестирования: выбор вопросов, проверка ответов, итог.

Здесь только чистые функции — без HTTP и без базы данных. Раньше всё это
жило внутри обработчиков в main.py, а выбор вопросов был продублирован
в двух эндпоинтах почти дословно; проверить подсчёт результата можно было
только через HTTP-запросы.

Функции принимают любые объекты с нужными атрибутами (ORM-модель Question
или простой SimpleNamespace в тестах).
"""
from __future__ import annotations

import random

LEVEL_TEXT = {"high": "Высокий", "medium": "Средний", "low": "Низкий"}

# Доля низкочастотных (редких) слов в тесте и размер теста
LOW_FREQUENCY_SHARE = 0.3
TEST_SIZE = 20
MIN_QUESTIONS = 5


def question_options(question) -> list[str]:
    """Варианты ответа: верный первым, затем дистракторы."""
    options = [question.correct_answer]
    for distractor in (question.distractor_1, question.distractor_2, question.distractor_3):
        if distractor:
            options.append(distractor)
    return options


def build_question_payload(question, rng=random) -> dict:
    """Вопрос в том виде, в каком его можно отдать браузеру.

    Правильный ответ НЕ передаётся. Раньше отдавались и индекс верного
    варианта ("correct"), и target_word — а target_word совпадает с
    correct_answer, то есть ответ уходил клиенту дважды. Проверка ответа
    делается на сервере по значению из БД.
    """
    options = question_options(question)
    rng.shuffle(options)
    return {
        "id": question.id,
        "question": question.definition,
        "options": options,
        "frequency_type": question.frequency_type or "medium",
        "difficulty": question.difficulty or 5,
    }


def select_test_questions(questions: list, rng=random) -> list:
    """Набор вопросов для одного теста: около 70% частотных и 30% редких слов.

    Если редких не хватает, добор идёт из общего пула. Вопросы без пометки
    частотности считаются средними.
    """
    high = [q for q in questions if q.frequency_type == "high"]
    medium = [q for q in questions if q.frequency_type == "medium"]
    low = [q for q in questions if q.frequency_type == "low"]
    if not high and not medium and not low:
        medium = list(questions)

    target_total = min(TEST_SIZE, len(questions))
    target_low = max(1, int(target_total * LOW_FREQUENCY_SHARE))
    target_common = target_total - target_low

    selected = []
    common = high + medium
    if common:
        selected.extend(rng.sample(common, min(target_common, len(common))))
    if low:
        selected.extend(rng.sample(low, min(target_low, len(low))))

    if len(selected) < target_total:
        chosen = {id(q) for q in selected}
        remaining = [q for q in questions if id(q) not in chosen]
        need = target_total - len(selected)
        selected.extend(rng.sample(remaining, min(need, len(remaining))))

    rng.shuffle(selected)
    return selected


def grade_answers(questions_by_id: dict, submitted: list) -> list[dict]:
    """Сверка ответов ученика с эталоном из БД.

    Клиент присылает только выбранный вариант: вердикт, частотность и
    сложность берутся из вопроса, а не из запроса. Раньше вердикт присылал
    браузер, и результат теста подделывался из DevTools.
    """
    graded = []
    for answer in submitted:
        question = questions_by_id[answer.question_id]
        given = (answer.answer or "").strip().lower()
        expected = (question.correct_answer or "").strip().lower()
        graded.append({
            "question_id": question.id,
            "is_correct": bool(given) and given == expected,
            "user_answer": answer.answer,
            "correct_answer": question.correct_answer,
            "frequency_type": question.frequency_type or "medium",
            "difficulty": question.difficulty or 5,
        })
    return graded


def calculate_level(percentage: float, high_freq_pct: float, low_freq_pct: float) -> str:
    """Уровень по результатам: высокий требует и общего результата,
    и владения редкими словами."""
    if percentage >= 90 and low_freq_pct >= 70:
        return "high"
    if percentage >= 70:
        return "medium"
    return "low"


def get_recommendation(level: str, grade: int, percentage: float) -> str:
    if level == "high":
        return (f"Отлично! Вы показали высокий уровень владения лексикой ({percentage:.0f}%). "
                f"Рекомендуем перейти к изучению слов {grade + 1} класса.")
    if level == "medium":
        return (f"Хороший результат ({percentage:.0f}%). Вы знаете большинство слов {grade} класса. "
                f"Рекомендуем уделить внимание редким словам.")
    return f"Результат: {percentage:.0f}%. Рекомендуем повторить основные слова {grade} класса."


def max_difficulty_reached(graded: list[dict], start: int = 5) -> int:
    """Адаптивная сложность: +1 после трёх верных подряд, -1 после двух
    неверных подряд, в пределах 1..10. Возвращает максимум за тест."""
    current = best = start
    streak_right = streak_wrong = 0
    for answer in graded:
        if answer["is_correct"]:
            streak_right, streak_wrong = streak_right + 1, 0
            if streak_right >= 3:
                current, streak_right = min(10, current + 1), 0
        else:
            streak_wrong, streak_right = streak_wrong + 1, 0
            if streak_wrong >= 2:
                current, streak_wrong = max(1, current - 1), 0
        best = max(best, current)
    return best


def summarize(graded: list[dict], grade: int) -> dict:
    """Итог теста: всё, что сохраняется в TestResult и отдаётся ученику."""
    total = len(graded)
    correct = sum(1 for a in graded if a["is_correct"])
    percentage = (correct / total * 100) if total else 0

    def count(freq: str, only_correct: bool = False) -> int:
        return sum(1 for a in graded
                   if a["frequency_type"] == freq and (a["is_correct"] or not only_correct))

    high_correct, high_total = count("high", True), count("high")
    medium_correct, medium_total = count("medium", True), count("medium")
    low_correct, low_total = count("low", True), count("low")

    high_pct = (high_correct / high_total * 100) if high_total else 100
    low_pct = (low_correct / low_total * 100) if low_total else 0
    level = calculate_level(percentage, high_pct, low_pct)

    return {
        "score": correct,
        "total": total,
        "percentage": percentage,
        "level": level,
        "max_difficulty": max_difficulty_reached(graded),
        "recommendation": get_recommendation(level, grade, percentage),
        "high": (high_correct, high_total, high_pct),
        "medium": (medium_correct, medium_total),
        "low": (low_correct, low_total, low_pct),
    }
