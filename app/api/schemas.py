"""Схемы входящих запросов."""
from pydantic import BaseModel


class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: str | None = None
    grade: int | None = 6


class UserLogin(BaseModel):
    username: str
    password: str


class WordInput(BaseModel):
    word: str


class QuestionCreate(BaseModel):
    target_word: str
    definition: str
    correct_answer: str
    distractor_1: str | None = None
    distractor_2: str | None = None
    distractor_3: str | None = None
    word_class: int = 6
    frequency_type: str = "medium"
    difficulty: int = 5
    part_of_speech: str | None = None


class TestStartRequest(BaseModel):
    grade: int = 6


class SubmittedAnswer(BaseModel):
    """Ответ ученика на один вопрос.

    Здесь намеренно НЕТ поля is_correct: раньше вердикт присылал браузер,
    и сервер принимал его на веру — результат теста подделывался из
    DevTools. Теперь сервер сверяет answer с эталоном из БД сам.
    """
    question_id: int
    answer: str | None = None  # текст выбранного варианта; None — без ответа


class TestCompleteRequest(BaseModel):
    # Идентификатор попытки из ответа на старт теста. Класс берётся из
    # попытки, а не из запроса: клиенту не доверяется ничего, кроме
    # выбранных вариантов.
    attempt_id: str
    answers: list[SubmittedAnswer]
