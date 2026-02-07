from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # Уровень ученика
    grade = Column(Integer, default=6)  # Класс ученика (4-7)
    current_level = Column(String(20), default="medium")  # low, medium, high
    
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_admin": self.is_admin,
            "grade": self.grade,
            "current_level": self.current_level
        }


class Question(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    target_word = Column(String(100), nullable=False, index=True)
    definition = Column(Text, nullable=False)
    correct_answer = Column(String(100), nullable=False)
    distractor_1 = Column(String(100), nullable=True)
    distractor_2 = Column(String(100), nullable=True)
    distractor_3 = Column(String(100), nullable=True)
    
    # Классификация слова
    word_class = Column(Integer, default=6)  # Для какого класса (4-7)
    frequency_type = Column(String(20), default="medium")  # high, medium, low
    difficulty = Column(Integer, default=5)  # 1-10 сложность
    
    part_of_speech = Column(String(20), nullable=True)
    is_approved = Column(Boolean, default=False)
    generation_log = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, nullable=True)
    
    def to_dict(self):
        options = [self.correct_answer]
        if self.distractor_1:
            options.append(self.distractor_1)
        if self.distractor_2:
            options.append(self.distractor_2)
        if self.distractor_3:
            options.append(self.distractor_3)
        
        return {
            "id": self.id,
            "question": self.definition,
            "options": options,
            "correct": 0,
            "target_word": self.target_word,
            "word_class": self.word_class,
            "frequency_type": self.frequency_type,
            "difficulty": self.difficulty,
            "part_of_speech": self.part_of_speech
        }


class TestResult(Base):
    __tablename__ = "test_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Результаты
    score = Column(Integer, nullable=False)
    total_questions = Column(Integer, nullable=False)
    percentage = Column(Float, nullable=False)
    
    # Детали по частотности
    high_freq_correct = Column(Integer, default=0)
    high_freq_total = Column(Integer, default=0)
    medium_freq_correct = Column(Integer, default=0)
    medium_freq_total = Column(Integer, default=0)
    low_freq_correct = Column(Integer, default=0)
    low_freq_total = Column(Integer, default=0)
    
    # Оценка
    grade_tested = Column(Integer, nullable=False)  # Какой класс тестировали
    level_achieved = Column(String(20), nullable=False)  # low, medium, high
    max_difficulty_reached = Column(Integer, default=5)
    
    # Рекомендация
    recommendation = Column(Text, nullable=True)
    
    completed_at = Column(DateTime, default=datetime.utcnow)


class TestAnswer(Base):
    """Ответы на отдельные вопросы теста"""
    __tablename__ = "test_answers"
    
    id = Column(Integer, primary_key=True, index=True)
    test_result_id = Column(Integer, nullable=False, index=True)
    question_id = Column(Integer, nullable=False)
    
    is_correct = Column(Boolean, nullable=False)
    user_answer = Column(String(100), nullable=True)
    correct_answer = Column(String(100), nullable=True)
    
    difficulty_at_answer = Column(Integer, default=5)
    
    answered_at = Column(DateTime, default=datetime.utcnow)


class GenerationLog(Base):
    __tablename__ = "generation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, nullable=True)
    step = Column(String(50), nullable=False)
    input_data = Column(Text, nullable=True)
    output_data = Column(Text, nullable=True)
    llm_prompt = Column(Text, nullable=True)
    llm_response = Column(Text, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)