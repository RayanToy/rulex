# -*- coding: utf-8 -*-
import os
import json
from typing import List, Optional
from anthropic import Anthropic
from dotenv import load_dotenv
import pymorphy3

load_dotenv()

morph = pymorphy3.MorphAnalyzer()


class QuestionGenerator:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        self.generation_log = []
    
    def _get_part_of_speech(self, word: str) -> Optional[str]:
        parsed = morph.parse(word)
        if not parsed:
            return None
        return parsed[0].tag.POS
    
    def _call_llm(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    
    def get_distractors(self, word: str) -> List[str]:
        """Получение дистракторов"""
        pos = self._get_part_of_speech(word)
        pos_map = {
        "NOUN": "существительное",
        "VERB": "глагол",
        "INFN": "инфинитив",
        "ADJF": "прилагательное"
        }
        pos_rus = pos_map.get(pos, "существительное")
    
        prompt = f"""Напиши 8 слов, связанных с темой слова '{word}' ({pos_rus}).

Требования:
- НЕ синонимы слова '{word}'
- НЕ однокоренные со словом '{word}'
- Из той же предметной области, но с ДРУГИМ значением
- Одно слово каждое
- Та же часть речи ({pos_rus})

Например, для слова "врач" подойдут: пациент, больница, лекарство (связаны по теме, но не синонимы).

Напиши только слова через запятую, без пояснений:"""
    
        response = self._call_llm(prompt)
    
        # Убираем вводный текст если есть
        if ':' in response:
            response = response.split(':', 1)[-1]
    
        response = response.strip().strip('.').strip()
    
        words = [w.strip().lower().rstrip('.').rstrip(',') for w in response.split(',')]
    
        valid = []
        for w in words:
            w = w.strip()
            if not w or w == word.lower() or '-' in w or len(w) < 2:
                continue
            if not w.isalpha():
                continue
            w_pos = self._get_part_of_speech(w)
            if w_pos == pos:
                valid.append(w)
        return valid[:3]
    
    def get_definition(self, word: str, distractors: List[str]) -> str:
        pos = self._get_part_of_speech(word)
        pos_map = {
            "NOUN": "существительное",
            "VERB": "глагол",
            "INFN": "инфинитив",
            "ADJF": "прилагательное"
        }
        pos_rus = pos_map.get(pos, "существительное")
        
        forbidden = ", ".join([word] + distractors)
        
        prompt = f"Напиши краткое толкование слова '{word}' ({pos_rus}). Не используй слова: {forbidden}. Только толкование, без слова и тире:"
        
        definition = self._call_llm(prompt)
        
        definition = definition.strip().lstrip('-').lstrip('—').strip()
        
        if definition and definition[0].islower():
            definition = definition[0].upper() + definition[1:]
        
        return definition
    
    def generate_question(self, word: str, student_class: int = 6) -> dict:
        word = word.lower().strip()
        
        distractors = self.get_distractors(word)
        definition = self.get_definition(word, distractors)
        
        definition = definition.rstrip('.')
        if not definition.endswith('—') and not definition.endswith('-'):
            definition = definition + ' —'
        
        return {
            "target_word": word,
            "definition": definition,
            "correct_answer": word,
            "distractors": distractors,
            "part_of_speech": self._get_part_of_speech(word),
            "word_class": student_class,
            "generation_log": "{}"
        }