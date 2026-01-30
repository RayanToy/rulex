from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
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
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_admin": self.is_admin
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
    word_class = Column(Integer, default=6)
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
            "part_of_speech": self.part_of_speech
        }


class TestResult(Base):
    __tablename__ = "test_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    score = Column(Integer, nullable=False)
    total_questions = Column(Integer, nullable=False)
    completed_at = Column(DateTime, default=datetime.utcnow)


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