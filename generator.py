# -*- coding: utf-8 -*-
import os
import json
import re
from typing import List, Optional, Tuple
from anthropic import Anthropic
from dotenv import load_dotenv
import pymorphy3
import csv
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

# ── Эвристики для отсева артефактов ──────────────────────────────────────
MIN_WORD_LEN = 3
MAX_WORD_LEN = 30
MIN_VOWEL_RATIO = 0.25
VOWELS = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')

# Запрещённые сочетания согласных (характерны для артефактов)
INVALID_CONSONANT_CLUSTERS = [
    'пкн', 'тпт', 'рьщ', 'гнн', 'бщт', 'вщр',
]


class WordListManager:
    """Управление списками слов по классам и словарём Шарова (оптимизировано, без pandas)"""
    
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
                self.freq_lists[class_num] = {}
                try:
                    with open(freq_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if not row:
                                continue
                            word = row[0].strip().lower()
                            # Пропускаем пустые строки, NaN и возможные заголовки
                            if not word or word == 'nan' or word == 'word' or word == 'слово':
                                continue
                            
                            freq = 1.0
                            if len(row) > 1 and row[1].strip():
                                try:
                                    freq = float(row[1].strip())
                                except ValueError:
                                    pass  # Если не получилось преобразовать в число, оставляем 1.0
                            
                            self.freq_lists[class_num][word] = freq
                    print(f"[OK] Freq class {class_num}: {len(self.freq_lists[class_num])} words")
                except Exception as e:
                    print(f"[ERROR] Freq class {class_num}: {e}")
            
            # Относительный список
            rel_path = DATA_DIR / f"class_{class_num}_relative.csv"
            if rel_path.exists():
                self.relative_lists[class_num] = set()
                try:
                    with open(rel_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if not row:
                                continue
                            word = row[0].strip().lower()
                            if not word or word == 'nan' or word == 'word' or word == 'слово':
                                continue
                            
                            self.relative_lists[class_num].add(word)
                    print(f"[OK] Relative class {class_num}: {len(self.relative_lists[class_num])} words")
                except Exception as e:
                    print(f"[ERROR] Relative class {class_num}: {e}")
    
    def _load_sharov(self):
        sharov_path = DATA_DIR / "sharov.csv"
        if not sharov_path.exists():
            return
        
        try:
            with open(sharov_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Пытаемся угадать разделитель (таб, точка с запятой или запятая)
                    if '\t' in line:
                        parts = line.split('\t')
                    elif ';' in line:
                        parts = line.split(';')
                    elif ',' in line:
                        parts = line.split(',')
                    else:
                        parts = line.split()
                        
                    if len(parts) >= 2:
                        word = parts[0].strip().lower()
                        if not word or word == 'nan' or word == 'word':
                            continue
                        try:
                            freq = float(parts[1].strip())
                            self.sharov[word] = freq
                        except ValueError:
                            continue  # Пропускаем заголовки или битые числа
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
    
    def _is_artifact(self, word: str) -> Tuple[bool, str]:
        """
        Эвристическая проверка: является ли слово артефактом (мусором из датасета).
        Работает без LLM, поэтому очень быстрая.
        """
        w = word.lower().strip()
        
        # Слишком короткое
        if len(w) < MIN_WORD_LEN:
            return True, f"Слишком короткое ({len(w)} букв)"
        
        # Слишком длинное
        if len(w) > MAX_WORD_LEN:
            return True, f"Слишком длинное ({len(w)} букв)"
        
        # Недостаточно гласных — признак артефакта (пкно, тпт, рьщ...)
        vowel_count = sum(1 for c in w if c in VOWELS)
        vowel_ratio = vowel_count / len(w) if len(w) > 0 else 0
        if vowel_ratio < MIN_VOWEL_RATIO:
            return True, f"Мало гласных ({vowel_ratio:.0%}), вероятно артефакт"
        
        # Запрещённые кластеры согласных
        for cluster in INVALID_CONSONANT_CLUSTERS:
            if cluster in w:
                return True, f"Недопустимое сочетание букв '{cluster}'"
        
        # pymorphy3 не знает слово совсем (score < 0.05 означает полную неизвестность)
        parsed = morph.parse(w)
        if parsed:
            best = parsed[0]
            if best.score < 0.05:
                return True, f"Слово неизвестно морфологическому словарю (score={best.score:.3f})"
        
        return False, "OK"
    
    def _filter_real_words_batch(self, words: List[str], batch_size: int = 30) -> List[str]:
        """
        Проверяет сразу пачку слов одним вызовом LLM.
        Отсеивает выдуманные слова типа 'травие', 'восьмибрат', 'плэда'.
        """
        real_words = []
        
        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            words_str = ', '.join(batch)
            
            prompt = f"""Ты эксперт русского языка и лексикограф.

Из списка слов выбери ТОЛЬКО те, которые реально существуют в стандартном русском языке и есть в словарях (Ожегов, РАС, Викисловарь).

ОТСЕИВАЙ слова которые:
- Выдуманы или являются артефактами датасета
- Звучат похоже на реальные, но НЕ существуют в словарях
- Являются ошибочными или искажёнными формами реальных слов

Примеры ВЫМЫШЛЕННЫХ слов которые нужно отсеять:
- травие (звучит как "трава" но это НЕ слово)
- восьмибрат (выдуманное)
- плэда (неправильное написание)
- будрить (похоже на "будить" но это НЕ слово)
- сугибнуть (выдуманное)
- ведрик (похоже на уменьшительное от "ведро" но такого слова НЕТ)
- пкно, тпеть, рьщарить (явный мусор)

Список для проверки:
{words_str}

Напиши ТОЛЬКО реальные существующие слова через запятую, без пояснений и нумерации:"""

            try:
                response = self._call_llm(prompt, max_tokens=200)
                
                # Парсим ответ
                if ':' in response:
                    response = response.split(':', 1)[-1]
                
                confirmed = [
                    w.strip().lower().rstrip('.').rstrip(',')
                    for w in response.split(',')
                    if w.strip()
                ]
                
                # Оставляем только те что были в батче И подтверждены
                batch_lower = [w.lower() for w in batch]
                confirmed_set = set(confirmed)
                
                valid_in_batch = [w for w in batch if w.lower() in confirmed_set]
                rejected = [w for w in batch if w.lower() not in confirmed_set]
                
                if rejected:
                    print(f"[FAKE] Отсеяно {len(rejected)} вымышленных слов: {', '.join(rejected[:5])}")
                
                real_words.extend(valid_in_batch)
                
            except Exception as e:
                print(f"[ERROR] Batch check failed: {e}")
                # При ошибке LLM — добавляем весь батч (лучше пропустить чем потерять)
                real_words.extend(batch)
        
        return real_words
    
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
        """
        Автоматическая генерация вопросов для класса.
        
        Изменения:
        1. Эвристическая фильтрация артефактов (быстро, без LLM)
        2. Батч-проверка реальности слов через LLM (экономия ~95% вызовов)
        3. Увеличенный пул кандидатов
        """
        
        all_words = self.word_manager.get_words_for_class(word_class)
        
        if not all_words:
            raise ValueError(f"Нет списка слов для {word_class} класса")
        
        # ── Шаг 1: эвристическая фильтрация (бесплатно) ─────────────────
        clean_words = []
        artifact_count = 0
        
        for w in all_words:
            # Базовая проверка
            is_valid, _ = self._is_basic_valid(w)
            if not is_valid:
                artifact_count += 1
                continue
            
            # Эвристика артефактов
            is_art, _ = self._is_artifact(w)
            if is_art:
                artifact_count += 1
                continue
            
            clean_words.append(w)
        
        print(f"[INFO] Class {word_class}: {len(all_words)} всего, {artifact_count} артефактов удалено, {len(clean_words)} чистых слов")
        
        if len(clean_words) < 5:
            raise ValueError(
                f"Недостаточно валидных слов для {word_class} класса "
                f"(после фильтрации осталось {len(clean_words)})"
            )
        
        # ── Шаг 2: батч-проверка реальности через LLM ────────────────────
        random.shuffle(clean_words)
        
        # Берём пул с запасом для LLM-проверки (count * 5 = 100 слов)
        # Это займёт примерно 100/30 ≈ 4 вызова LLM вместо 100
        check_pool_size = min(len(clean_words), count * 5)
        check_pool = clean_words[:check_pool_size]
        
        print(f"[INFO] Проверяю реальность {len(check_pool)} слов через LLM...")
        real_words = self._filter_real_words_batch(check_pool, batch_size=30)
        
        print(f"[INFO] После LLM-проверки: {len(real_words)} реальных слов")
        
        # ── Шаг 3: если мало — добираем из оставшихся ────────────────────
        if len(real_words) < count * 2:
            print(f"[INFO] Мало реальных слов, проверяю дополнительный батч...")
            extra_pool = clean_words[check_pool_size : check_pool_size + count * 3]
            if extra_pool:
                extra_real = self._filter_real_words_batch(extra_pool, batch_size=30)
                real_words.extend(extra_real)
                print(f"[INFO] После дополнительной проверки: {len(real_words)} реальных слов")
        
        if len(real_words) < 5:
            raise ValueError(f"Критически мало реальных слов: {len(real_words)}")
        
        # ── Шаг 4: распределение частотности ─────────────────────────────
        freq_distribution = (
            ["high"] * int(count * 0.4) +
            ["medium"] * int(count * 0.4) +
            ["low"] * int(count * 0.2)
        )
        random.shuffle(freq_distribution)
        
        # ── Шаг 5: основной цикл генерации ───────────────────────────────
        # Берём до 8× слов на случай отсева при генерации
        candidates = real_words[:min(len(real_words), count * 8)]
        
        questions = []
        
        for word in candidates:
            if len(questions) >= count:
                break
            
            # Проверка пригодности для теста (специальные термины и т.д.)
            is_suitable, reason = self._check_word_suitability(word)
            if not is_suitable:
                print(f"[SKIP] {word}: {reason}")
                continue
            
            try:
                freq_type = (
                    freq_distribution[len(questions)]
                    if len(questions) < len(freq_distribution)
                    else "medium"
                )
                question = self.generate_question(word, word_class, freq_type)
                questions.append(question)
                print(f"[OK] {len(questions)}/{count}: {word}")
            except Exception as e:
                print(f"[ERROR] {word}: {e}")
                continue
        
        # ── Шаг 6: проверка результата ───────────────────────────────────
        if len(questions) < count:
            print(
                f"[WARN] Удалось сгенерировать только {len(questions)}/{count} вопросов. "
                f"Проверьте качество словаря class_{word_class}_relative.csv"
            )
        
        if len(questions) < max(5, count // 2):
            raise ValueError(
                f"Критически мало вопросов: {len(questions)}/{count}. "
                f"Словарь класса {word_class} содержит слишком много артефактов или специальных терминов."
            )
        
        return questions
    
    def has_word_list(self, word_class: int) -> bool:
        return word_class in self.word_manager.relative_lists and len(self.word_manager.relative_lists[word_class]) > 0
    
    def get_available_classes(self) -> List[int]:
        return sorted(self.word_manager.relative_lists.keys())