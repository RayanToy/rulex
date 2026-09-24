"""Логика тестирования без HTTP и без базы.

Раньше эти правила жили внутри обработчиков, и проверить их можно было
только сквозным запросом. Теперь это чистые функции, и граничные случаи
проверяются напрямую: нехватка редких слов, повторы вопросов, адаптивная
сложность на краях шкалы.
"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services import assessment  # noqa: E402


def make_question(qid, freq="medium", difficulty=5, answer=None, distractors=("а", "б", "в")):
    answer = answer or f"слово{qid}"
    d1, d2, d3 = (list(distractors) + [None, None, None])[:3]
    return SimpleNamespace(id=qid, definition=f"толкование {qid}", correct_answer=answer,
                           distractor_1=d1, distractor_2=d2, distractor_3=d3,
                           frequency_type=freq, difficulty=difficulty)


def bank(high=0, medium=0, low=0, untagged=0):
    questions, qid = [], 0
    for freq, n in (("high", high), ("medium", medium), ("low", low), (None, untagged)):
        for _ in range(n):
            qid += 1
            questions.append(make_question(qid, freq))
    return questions


class TestPayload:
    def test_answer_is_not_sent_to_client(self):
        payload = assessment.build_question_payload(make_question(1), rng=random.Random(0))
        assert set(payload) == {"id", "question", "options", "frequency_type", "difficulty"}

    def test_options_are_answer_plus_distractors(self):
        q = make_question(1, answer="кот", distractors=("стул", "окно", "ветер"))
        payload = assessment.build_question_payload(q, rng=random.Random(0))
        assert sorted(payload["options"]) == sorted(["кот", "стул", "окно", "ветер"])

    def test_empty_distractors_are_skipped(self):
        q = make_question(1, answer="кот", distractors=("стул", "", None))
        assert assessment.question_options(q) == ["кот", "стул"]

    def test_missing_metadata_gets_defaults(self):
        q = make_question(1, freq=None, difficulty=None)
        payload = assessment.build_question_payload(q, rng=random.Random(0))
        assert payload["frequency_type"] == "medium"
        assert payload["difficulty"] == 5


class TestSelection:
    @staticmethod
    def select(questions, seed=0):
        return assessment.select_test_questions(questions, rng=random.Random(seed))

    def test_seventy_thirty_split(self):
        selected = self.select(bank(high=15, medium=15, low=10))
        assert len(selected) == assessment.TEST_SIZE
        assert sum(q.frequency_type == "low" for q in selected) == 6   # int(20 * 0.3)

    @pytest.mark.parametrize("sizes", [
        dict(high=15, medium=15, low=10),
        dict(high=25),                  # редких нет — добор из частотных
        dict(low=25),                   # частотных нет — добор из редких
        dict(high=3, low=2),            # банк меньше размера теста
        dict(untagged=12),              # частотность не размечена
        dict(medium=2, low=1, untagged=9),
    ])
    def test_no_repeats_and_right_size(self, sizes):
        questions = bank(**sizes)
        for seed in range(20):
            selected = self.select(questions, seed)
            assert len({q.id for q in selected}) == len(selected), "вопрос повторился"
            assert len(selected) == min(assessment.TEST_SIZE, len(questions))

    def test_seeded_rng_is_reproducible(self):
        questions = bank(high=15, medium=15, low=10)
        assert [q.id for q in self.select(questions, 3)] == [q.id for q in self.select(questions, 3)]


class TestGrading:
    @staticmethod
    def answer(qid, text):
        return SimpleNamespace(question_id=qid, answer=text)

    def test_case_and_spaces_do_not_matter(self):
        q = make_question(1, answer="Кот")
        graded = assessment.grade_answers({1: q}, [self.answer(1, "  кот ")])
        assert graded[0]["is_correct"] is True

    def test_empty_answer_is_wrong(self):
        q = make_question(1, answer="кот")
        for given in ("", "   ", None):
            assert assessment.grade_answers({1: q}, [self.answer(1, given)])[0]["is_correct"] is False

    def test_metadata_comes_from_question_not_request(self):
        q = make_question(1, freq="low", difficulty=8, answer="кот")
        submitted = SimpleNamespace(question_id=1, answer="пёс", is_correct=True,
                                    frequency_type="high", difficulty=1)
        graded = assessment.grade_answers({1: q}, [submitted])[0]
        assert graded["is_correct"] is False
        assert (graded["frequency_type"], graded["difficulty"]) == ("low", 8)
        assert graded["correct_answer"] == "кот"


def graded_seq(pattern, freq="medium"):
    """'++-' -> верно, верно, неверно."""
    return [{"is_correct": c == "+", "frequency_type": freq} for c in pattern]


class TestAdaptiveDifficulty:
    @pytest.mark.parametrize("pattern, expected", [
        ("", 5),
        ("++", 5),               # двух верных мало
        ("+++", 6),
        ("++++++", 7),
        ("++-+", 5),             # серия прервана
        ("+++--", 6),            # спад после подъёма не отменяет максимум
        ("--" * 10, 5),          # ниже старта максимум не опускается
        ("+" * 60, 10),          # потолок шкалы
    ])
    def test_max_difficulty(self, pattern, expected):
        assert assessment.max_difficulty_reached(graded_seq(pattern)) == expected

    def test_floor_then_recovery(self):
        # 8 пар ошибок опускают с 5 до 1 (не ниже), 6 верных поднимают до 3
        graded = graded_seq("--" * 8 + "+" * 6)
        assert assessment.max_difficulty_reached(graded, start=5) == 5
        assert assessment.max_difficulty_reached(graded, start=1) == 3


class TestSummary:
    def test_counts_by_frequency(self):
        graded = graded_seq("++-", "high") + graded_seq("+-", "medium") + graded_seq("+--", "low")
        s = assessment.summarize(graded, grade=6)
        assert (s["score"], s["total"]) == (4, 8)
        assert s["percentage"] == 50
        assert s["high"][:2] == (2, 3)
        assert s["medium"] == (1, 2)
        assert s["low"][:2] == (1, 3)
        assert s["level"] == "low"

    def test_high_level_requires_rare_words(self):
        # 95% верных, но из редких — только половина
        graded = graded_seq("+" * 17, "high") + graded_seq("+-", "low") + graded_seq("+", "medium")
        s = assessment.summarize(graded, grade=6)
        assert s["percentage"] == 95
        assert s["level"] == "medium"

    def test_high_level(self):
        graded = graded_seq("+" * 14, "high") + graded_seq("+" * 6, "low")
        s = assessment.summarize(graded, grade=6)
        assert s["level"] == "high"
        assert "7 класса" in s["recommendation"]

    def test_empty_test(self):
        s = assessment.summarize([], grade=6)
        assert (s["score"], s["total"], s["percentage"], s["level"]) == (0, 0, 0, "low")
