# -*- coding: utf-8 -*-
import os
import json
import re
from typing import List, Optional, Tuple
from anthropic import Anthropic
from dotenv import load_dotenv
import pymorphy3
import pandas as pd
from pathlib import Path
import random

load_dotenv()

morph = pymorphy3.MorphAnalyzer()
DATA_DIR = Path(__file__).parent / "data"

# Стоп-слова (служебные, очень частые)
STOP_WORDS = {
    'и', 'в', 'на', 'с', 'к', 'по', 'за', 'из', 'о', 'у', 'от', 'для', 'при',
    'до', 'или', 'что', 'как', 'это', 'который', 'его', 'её', 'их', 'свой',
    'этот', 'тот', 'весь', 'сам', 'быть', 'а', 'но', 'да', 'не', 'ни', 'же',
    'бы', 'ли', 'вот', 'только', 'ещё', 'уже', 'где', 'когда', 'если', 'чтобы'
}


class WordListManager:
    """Управление списками слов по классам и словарём Шарова"""
    
    def __init__(self):
        self.freq_lists = {}
        self.relative_lists = {}
        self.sharov = {}
        self._load_all()
    
    def _load_all(self):
        self._load_class_lists()
        self._load_sharov()
    
    def _load_class_lists(self):
        for class_num in range(2, 12):
            # Частотный список
            freq_path = DATA_DIR / f"class_{class_num}_freq.csv"
            if freq_path.exists():
                try:
                    df = pd.read_csv(freq_path, encoding='utf-8')
                    word_col = df.columns[0]
                    freq_col = df.columns[1] if len(df.columns) > 1 else None
                    
                    if freq_col:
                        self.freq_lists[class_num] = {
                            str(row[word_col]).lower().strip(): float(row[freq_col]) if pd.notna(row[freq_col]) else 1
                            for _, row in df.iterrows()
                            if pd.notna(row[word_col]) and str(row[word_col]).strip()
                        }
                    else:
                        self.freq_lists[class_num] = {
                            str(w).lower().strip(): 1 for w in df[word_col].dropna()
                        }
                    print(f"[OK] Freq class {class_num}: {len(self.freq_lists[class_num])} words")
                except Exception as e:
                    print(f"[ERROR] Freq class {class_num}: {e}")
            
            # Относительный список
            rel_path = DATA_DIR / f"class_{class_num}_relative.csv"
            if rel_path.exists():
                try:
                    df = pd.read_csv(rel_path, encoding='utf-8')
                    word_col = df.columns[0]
                    self.relative_lists[class_num] = {
                        str(w).lower().strip() for w in df[word_col].dropna()
                        if str(w).strip() and str(w) != 'nan'
                    }
                    print(f"[OK] Relative class {class_num}: {len(self.relative_lists[class_num])} words")
                except Exception as e:
                    print(f"[ERROR] Relative class {class_num}: {e}")
    
    def _load_sharov(self):
        sharov_path = DATA_DIR / "sharov.csv"
        if not sharov_path.exists():
            return
        
        try:
            for sep in ['\t', ';', ',']:
                try:
                    df = pd.read_csv(sharov_path, sep=sep, encoding='utf-8', on_bad_lines='skip')
                    if len(df.columns) >= 2:
                        break
                except:
                    continue
            
            if len(df.columns) >= 2:
                word_col = df.columns[0]
                freq_col = df.columns[1]
                
                for _, row in df.iterrows():
                    word = str(row[word_col]).lower().strip()
                    try:
                        freq = float(row[freq_col])
                        if word and word != 'nan':
                            self.sharov[word] = freq
                    except:
                        continue
                print(f"[OK] Sharov: {len(self.sharov)} words")
        except Exception as e:
            print(f"[ERROR] Sharov: {e}")
    
    def get_word_frequency_in_class(self, word: str, class_num: int) -> float:
        word = word.lower().strip()
        if class_num in self.freq_lists:
            return self.freq_lists[class_num].get(word, 0)
        return 0
    
    def get_total_frequency_below_class(self, word: str, target_class: int) -> float:
        word = word.lower().strip()
        total = 0
        for class_num in range(2, target_class):
            if class_num in self.freq_lists:
                total += self.freq_lists[class_num].get(word, 0)
        return total
    
    def word_first_appears_in_class(self, word: str) -> Optional[int]:
        word = word.lower().strip()
        for class_num in range(2, 12):
            if class_num in self.relative_lists:
                if word in self.relative_lists[class_num]:
                    return class_num
        return None
    
    def get_sharov_frequency(self, word: str) -> float:
        return self.sharov.get(word.lower().strip(), 0)
    
    def is_word_known_below_class(self, word: str, target_class: int) -> bool:
        first_class = self.word_first_appears_in_class(word)
        if first_class is not None and first_class < target_class:
            return True
        return self.get_sharov_frequency(word) >= 50
    
    def get_words_for_class(self, class_num: int) -> List[str]:
        if class_num in self.relative_lists:
            return list(self.relative_lists[class_num])
        return []


class QuestionGenerator:
    """Генератор вопросов для теста на словарный запас"""
    
    # Контекст для LLM о назначении теста
    SYSTEM_CONTEXT = """Ты помогаешь создавать тестовые задания для проверки СЛОВАРНОГО ЗАПАСА школьников.

Цель теста — проверить, знает ли ученик ЗНАЧЕНИЕ слова, а не специальные знания.

ВАЖНЫЕ ПРАВИЛА:
1. НЕ использовать территориальные слова (названия городов, регионов, стран)
2. НЕ использовать этнонимы (названия народов, национальностей)
3. НЕ использовать узкоспециальные термины (медицинские, юридические, технические)
4. НЕ использовать имена собственные
5. НЕ использовать региональные/диалектные слова
6. Слова должны быть общеупотребительными в русском языке
7. Значение слова должно быть понятно из общего образования, а не из специальных знаний"""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        self.word_manager = WordListManager()
        self.generation_log = []
    
    def _log(self, step: str, data: dict):
        self.generation_log.append({"step": step, **data})
    
    def _get_pos(self, word: str) -> Optional[str]:
        parsed = morph.parse(word)
        if not parsed:
            return None
        return parsed[0].tag.POS
    
    def _get_lemma(self, word: str) -> str:
        parsed = morph.parse(word)
        if parsed:
            return parsed[0].normal_form
        return word.lower()
    
    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.SYSTEM_CONTEXT,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    
    def _check_word_suitability(self, word: str) -> Tuple[bool, str]:
        """Проверка слова через LLM на пригодность для теста словарного запаса"""
        
        prompt = f"""Оцени, подходит ли слово "{word}" для теста на СЛОВАРНЫЙ ЗАПАС школьника.

Слово НЕ подходит, если это:
- Название места (город, страна, регион, река, гора)
- Название народа или национальности (латыш, немец, татарин)
- Прилагательное от географического названия (московский, тверской, балтийский)
- Узкоспециальный термин (медицинский, юридический, технический)
- Устаревшее или диалектное слово
- Имя собственное
- Слово, требующее специальных знаний для понимания

Слово ПОДХОДИТ, если это общеупотребительное слово, значение которого можно объяснить без специальных знаний.

Ответь СТРОГО в формате:
ПОДХОДИТ: да/нет
ПРИЧИНА: краткое объяснение"""

        try:
            response = self._call_llm(prompt, max_tokens=100)
            
            is_suitable = "ПОДХОДИТ: да" in response.lower() or "подходит: да" in response.lower()
            
            # Извлекаем причину
            reason = "OK" if is_suitable else "Не подходит для теста"
            if "ПРИЧИНА:" in response:
                reason = response.split("ПРИЧИНА:")[-1].strip()
            
            self._log("word_check", {"word": word, "suitable": is_suitable, "reason": reason})
            
            return is_suitable, reason
        except Exception as e:
            self._log("word_check_error", {"word": word, "error": str(e)})
            return True, "Не удалось проверить"
    
    def _is_basic_valid(self, word: str) -> Tuple[bool, str]:
        """Базовая проверка слова (без LLM)"""
        word = word.lower().strip()
        
        if not word or len(word) < 2:
            return False, "Слишком короткое"
        
        if not word.isalpha():
            return False, "Содержит не-буквы"
        
        if '-' in word:
            return False, "Содержит дефис"
        
        pos = self._get_pos(word)
        if pos not in ['NOUN', 'VERB', 'INFN']:
            return False, f"Неподходящая часть речи: {pos}"
        
        parsed = morph.parse(word)
        if parsed:
            tags = str(parsed[0].tag)
            if 'Abbr' in tags or 'NUMB' in tags:
                return False, "Аббревиатура или число"
        
        return True, "OK"
    
    def _get_distractors(self, word: str, target_class: int) -> List[str]:
        """Получение дистракторов"""
        pos = self._get_pos(word)
        pos_rus = {'NOUN': 'существительное', 'VERB': 'глагол', 'INFN': 'инфинитив'}.get(pos, 'существительное')
        
        prompt = f"""Для теста на словарный запас нужны слова-дистракторы к слову "{word}" ({pos_rus}).

Требования к дистракторам:
1. Должны быть гиперонимами или гипонимами слова "{word}"
2. НЕ синонимы слова "{word}" (значение должно быть ДРУГИМ)
3. НЕ однокоренные со словом "{word}"
4. Та же часть речи ({pos_rus})
5. Общеупотребительные слова (не специальные термины)
6. НЕ географические названия, НЕ этнонимы
7. Одно слово каждое, без дефисов

Напиши 10 подходящих слов через запятую, без пояснений:"""
        
        response = self._call_llm(prompt, max_tokens=150)
        self._log("distractors_response", {"word": word, "response": response})
        
        # Парсинг
        if ':' in response:
            response = response.split(':', 1)[-1]
        
        candidates = [w.strip().lower().rstrip('.').rstrip(',') for w in response.split(',')]
        
        valid_distractors = []
        target_pos = self._get_pos(word)
        
        for candidate in candidates:
            if not candidate or len(candidate) < 2:
                continue
            if not candidate.isalpha() or '-' in candidate:
                continue
            if candidate == word.lower():
                continue
            
            # Проверка части речи
            cand_pos = self._get_pos(candidate)
            if cand_pos != target_pos:
                continue
            
            valid_distractors.append(candidate)
            
            if len(valid_distractors) >= 3:
                break
        
        return valid_distractors
    
    def _get_definition(self, word: str, distractors: List[str], target_class: int) -> str:
        """Получение толкования"""
        pos = self._get_pos(word)
        pos_rus = {'NOUN': 'существительное', 'VERB': 'глагол', 'INFN': 'инфинитив'}.get(pos, 'существительное')
        
        forbidden = ', '.join([word] + distractors)
        
        prompt = f"""Напиши толкование для слова "{word}" ({pos_rus}) для теста на словарный запас школьника.

ВАЖНО:
1. Толкование должно быть кратким и понятным
2. НЕ используй слова: {forbidden} и однокоренные им
3. НЕ используй специальные термины
4. НЕ ссылайся на географические названия или национальности
5. Используй только простые, общеупотребительные слова
6. Толкование должно однозначно указывать на слово "{word}"

Напиши только само толкование, без целевого слова и тире:"""
        
        definition = self._call_llm(prompt, max_tokens=150)
        definition = definition.strip().lstrip('-—').strip()
        
        self._log("definition_initial", {"word": word, "definition": definition})
        
        # Проверка и корректировка
        for attempt in range(3):
            # Проверяем на наличие запрещённых слов
            needs_correction = False
            
            words_in_def = re.findall(r'[а-яёА-ЯЁ]+', definition.lower())
            for w in words_in_def:
                lemma = self._get_lemma(w)
                # Проверяем однокоренные
                for forbidden_word in [word] + distractors:
                    if lemma == self._get_lemma(forbidden_word):
                        needs_correction = True
                        break
                    # Простая проверка на общий корень
                    if len(lemma) > 3 and len(self._get_lemma(forbidden_word)) > 3:
                        if lemma[:4] == self._get_lemma(forbidden_word)[:4]:
                            needs_correction = True
                            break
            
            if not needs_correction:
                break
            
            # Корректировка
            correction_prompt = f"""Исправь толкование так, чтобы не использовать слова, однокоренные с: {forbidden}

Текущее толкование: "{definition}"

Сохрани смысл, но перефразируй. Напиши только исправленное толкование:"""
            
            definition = self._call_llm(correction_prompt, max_tokens=150)
            definition = definition.strip().lstrip('-—').strip()
            self._log("definition_corrected", {"attempt": attempt + 1, "definition": definition})
        
        return definition
    
    def generate_question(self, word: str, word_class: int = 6, frequency_type: str = "medium") -> dict:
        """Генерация одного вопроса"""
        self.generation_log = []
        word = word.lower().strip()
        
        self._log("start", {"word": word, "word_class": word_class})
        
        # Базовая проверка
        is_valid, reason = self._is_basic_valid(word)
        if not is_valid:
            raise ValueError(f"Слово '{word}' не подходит: {reason}")
        
        # Получение дистракторов
        distractors = self._get_distractors(word, word_class)
        
        if len(distractors) < 2:
            raise ValueError(f"Не удалось найти дистракторы для '{word}'")
        
        # Получение толкования
        definition = self._get_definition(word, distractors, word_class)
        
        # Форматирование
        if definition and definition[0].islower():
            definition = definition[0].upper() + definition[1:]
        definition = definition.rstrip('.')
        if not definition.endswith('—') and not definition.endswith('-'):
            definition = definition + ' —'
        
        result = {
            "target_word": word,
            "definition": definition,
            "correct_answer": word,
            "distractors": distractors,
            "part_of_speech": self._get_pos(word),
            "word_class": word_class,
            "frequency_type": frequency_type,
            "generation_log": json.dumps(self.generation_log, ensure_ascii=False, default=str)
        }
        
        self._log("complete", {"success": True})
        return result
    
    def generate_questions_for_class(self, word_class: int, count: int = 20) -> List[dict]:
        """Автоматическая генерация вопросов для класса"""
        
        all_words = self.word_manager.get_words_for_class(word_class)
        
        if not all_words:
            raise ValueError(f"Нет списка слов для {word_class} класса")
        
        # Базовая фильтрация
        basic_valid = []
        for w in all_words:
            is_valid, _ = self._is_basic_valid(w)
            if is_valid:
                basic_valid.append(w)
        
        print(f"[INFO] Class {word_class}: {len(all_words)} total, {len(basic_valid)} basic valid")
        
        if len(basic_valid) < 5:
            raise ValueError(f"Недостаточно валидных слов для {word_class} класса")
        
        # Перемешиваем и берём с запасом
        random.shuffle(basic_valid)
        candidates = basic_valid[:count * 3]  # Берём в 3 раза больше на случай отсева
        
        # Распределение частотности
        freq_distribution = (
            ["high"] * int(count * 0.4) +
            ["medium"] * int(count * 0.4) +
            ["low"] * int(count * 0.2)
        )
        random.shuffle(freq_distribution)
        
        questions = []
        checked_words = 0
        
        for word in candidates:
            if len(questions) >= count:
                break
            
            checked_words += 1
            
            # Проверка через LLM (каждое 3-е слово для экономии)
            if checked_words % 3 == 1:
                is_suitable, reason = self._check_word_suitability(word)
                if not is_suitable:
                    print(f"[SKIP] {word}: {reason}")
                    continue
            
            try:
                freq_type = freq_distribution[len(questions)] if len(questions) < len(freq_distribution) else "medium"
                question = self.generate_question(word, word_class, freq_type)
                questions.append(question)
                print(f"[OK] {len(questions)}/{count}: {word}")
            except Exception as e:
                print(f"[ERROR] {word}: {e}")
                continue
        
        return questions
    
    def has_word_list(self, word_class: int) -> bool:
        return word_class in self.word_manager.relative_lists and len(self.word_manager.relative_lists[word_class]) > 0
    
    def get_available_classes(self) -> List[int]:
        return sorted(self.word_manager.relative_lists.keys())