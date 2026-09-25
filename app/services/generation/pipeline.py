"""QuestionGenerator: порядок этапов и генерация на уровне класса.

Сами этапы — в соседних модулях; здесь только то, в каком порядке они
вызываются, и журнал generation_log, который сохраняется вместе
с вопросом.
"""
import asyncio
import json
import os
import random

from anthropic import APIConnectionError, APIError, APIStatusError

from app.services.generation.console import log
from app.services.generation.definitions import DefinitionWriter
from app.services.generation.distractors import DistractorPicker
from app.services.generation.model_calls import ModelCaller
from app.services.generation.morphology import get_pos
from app.services.generation.realness import RealnessFilter, is_artifact, is_basic_valid
from app.services.generation.suitability import SuitabilityCheck
from app.services.llm import OllamaError
from app.services.wordlists import get_word_manager


class QuestionGenerator:
    """Генератор вопросов для теста на словарный запас"""

    def __init__(self, llm: ModelCaller | None = None, word_manager=None):
        # Клиент и словари общие на процесс, состояние генерации — своё:
        # generation_log и счётчики изменяемые, и общий экземпляр
        # генератора гонялся бы между параллельными запросами.
        self.llm = llm if llm is not None else ModelCaller()
        self.word_manager = word_manager if word_manager is not None else get_word_manager()
        self.generation_log: list[dict] = []

        self.realness = RealnessFilter(self.llm, self.word_manager)
        self.suitability = SuitabilityCheck(self.llm, trace=self._trace)
        self.distractors = DistractorPicker(self.llm, trace=self._trace)
        self.definitions = DefinitionWriter(self.llm, trace=self._trace)

    @property
    def client(self):
        return self.llm.client

    @property
    def model(self) -> str:
        return self.llm.model

    @property
    def structured_stats(self) -> dict:
        return self.llm.structured_stats

    def _trace(self, step: str, data: dict) -> None:
        self.generation_log.append({"step": step, **data})

    def generate_question(self, word: str, word_class: int = 6, frequency_type: str = "medium") -> dict:
        """Генерация одного вопроса"""
        self.generation_log = []
        word = word.lower().strip()

        self._trace("start", {"word": word, "word_class": word_class})

        # Базовая проверка
        is_valid, reason = is_basic_valid(word)
        if not is_valid:
            raise ValueError(f"Слово '{word}' не подходит: {reason}")

        # Получение дистракторов
        distractors = self.distractors.pick(word)

        if len(distractors) < 2:
            raise ValueError(f"Не удалось найти дистракторы для '{word}'")

        # Получение толкования
        definition = self.definitions.write(word, distractors)

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
            "part_of_speech": get_pos(word),
            "word_class": word_class,
            "frequency_type": frequency_type,
            "generation_log": json.dumps(self.generation_log, ensure_ascii=False, default=str)
        }

        self._trace("complete", {"success": True})
        return result

    def _prepare_candidates(self, word_class: int, count: int) -> tuple[list[str], list[str]]:
        """Шаги 1–4: эвристики, батч-проверка реальности, распределение частотности.

        Вынесено отдельно, чтобы синхронная и асинхронная генерация
        использовали ровно одну и ту же подготовку.
        """
        all_words = self.word_manager.get_words_for_class(word_class)

        if not all_words:
            raise ValueError(f"Нет списка слов для {word_class} класса")

        # ── Шаг 1: эвристическая фильтрация (бесплатно) ─────────────────
        clean_words = []
        artifact_count = 0

        for w in all_words:
            # Базовая проверка
            is_valid, _ = is_basic_valid(w)
            if not is_valid:
                artifact_count += 1
                continue

            # Эвристика артефактов
            is_art, _ = is_artifact(w)
            if is_art:
                artifact_count += 1
                continue

            clean_words.append(w)

        log(f"[INFO] Class {word_class}: {len(all_words)} всего, {artifact_count} артефактов удалено, {len(clean_words)} чистых слов")

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

        real_words = self.realness.confirm(check_pool)

        # ── Шаг 3: если мало — добираем из оставшихся ────────────────────
        if len(real_words) < count * 2:
            log("[INFO] Мало реальных слов, проверяю дополнительный батч...")
            extra_pool = clean_words[check_pool_size : check_pool_size + count * 3]
            if extra_pool:
                real_words.extend(self.realness.confirm(extra_pool))
                log(f"[INFO] После дополнительной проверки: {len(real_words)} реальных слов")

        if len(real_words) < 5:
            raise ValueError(f"Критически мало реальных слов: {len(real_words)}")

        # ── Шаг 4: распределение частотности ─────────────────────────────
        freq_distribution = (
            ["high"] * int(count * 0.4) +
            ["medium"] * int(count * 0.4) +
            ["low"] * int(count * 0.2)
        )
        random.shuffle(freq_distribution)

        candidates = real_words[:min(len(real_words), count * 8)]
        return candidates, freq_distribution

    def generate_questions_for_class(self, word_class: int, count: int = 20) -> list[dict]:
        """
        Автоматическая генерация вопросов для класса.

        Изменения:
        1. Эвристическая фильтрация артефактов (быстро, без LLM)
        2. Батч-проверка реальности слов через LLM (экономия ~95% вызовов)
        3. Увеличенный пул кандидатов
        """

        candidates, freq_distribution = self._prepare_candidates(word_class, count)

        questions = []

        for word in candidates:
            if len(questions) >= count:
                break

            # Проверка пригодности для теста (специальные термины и т.д.)
            is_suitable, reason = self.suitability.check(word)
            if not is_suitable:
                log(f"[SKIP] {word}: {reason}")
                continue

            try:
                freq_type = (
                    freq_distribution[len(questions)]
                    if len(questions) < len(freq_distribution)
                    else "medium"
                )
                question = self.generate_question(word, word_class, freq_type)
                questions.append(question)
                log(f"[OK] {len(questions)}/{count}: {word}")
            except Exception as e:
                log(f"[ERROR] {word}: {e}")
                continue

        # ── Шаг 6: проверка результата ───────────────────────────────────
        if len(questions) < count:
            log(
                f"[WARN] Удалось сгенерировать только {len(questions)}/{count} вопросов. "
                f"Проверьте качество словаря class_{word_class}_relative.csv"
            )

        if len(questions) < max(5, count // 2):
            raise ValueError(
                f"Критически мало вопросов: {len(questions)}/{count}. "
                f"Словарь класса {word_class} содержит слишком много артефактов или специальных терминов."
            )

        return questions

    def _worker(self) -> "QuestionGenerator":
        """Генератор для одной параллельной задачи.

        Свой экземпляр нужен потому, что generation_log и счётчики
        изменяемые: при параллельном запуске задачи затирали бы их друг
        у друга. Клиент, модель и словари при этом общие, так что создание
        экземпляра почти бесплатно.
        """
        return QuestionGenerator(llm=ModelCaller(self.llm.client, self.llm.model),
                                 word_manager=self.word_manager)

    def _make_one(self, word: str, word_class: int, freq_type: str) -> dict | None:
        """Одно задание целиком, в своём экземпляре генератора."""
        worker = self._worker()
        suitable, reason = worker.suitability.check(word)
        if not suitable:
            log(f"[SKIP] {word}: {reason}")
            return None
        try:
            return worker.generate_question(word, word_class, freq_type)
        except (ValueError, APIStatusError, APIConnectionError, APIError, OllamaError) as exc:
            log(f"[ERROR] {word}: {type(exc).__name__}")
            return None

    async def agenerate_questions_for_class(
        self, word_class: int, count: int = 20, concurrency: int | None = None
    ) -> list[dict]:
        """Асинхронная генерация: вызовы уходят из event loop в поток.

        Главное здесь — НЕ ускорение, а то, что обработчик перестаёт
        блокировать сервер. Синхронная версия делает 3–6 вызовов LLM
        на слово подряд, и на время генерации (минуты) приложение
        не отвечало никому.

        Про параллелизм. По умолчанию он ВЫКЛЮЧЕН (concurrency=1),
        и это результат замера, а не осторожность: на шести словах
        пять параллельных задач дали через шлюз router.cheap 251 секунду
        против 53 последовательных, то есть впятеро хуже — запросы
        троттлятся и SDK уходит в ретраи. На локальной Ollama выигрыша
        тоже нет: она обслуживает один GPU и выполняет запросы по очереди.
        Повышать RULEX_GEN_CONCURRENCY имеет смысл только против прямого
        API и только с замером — вслепую это делает хуже.
        """
        if concurrency is None:
            concurrency = max(1, int(os.getenv("RULEX_GEN_CONCURRENCY") or 1))
        candidates, freq_distribution = await asyncio.to_thread(
            self._prepare_candidates, word_class, count
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def one(word: str, freq_type: str) -> dict | None:
            async with semaphore:
                return await asyncio.to_thread(self._make_one, word, word_class, freq_type)

        questions: list[dict] = []
        pending = list(candidates)
        while pending and len(questions) < count:
            wave, pending = pending[:count - len(questions)], pending[count - len(questions):]
            tasks = [
                one(word, freq_distribution[(len(questions) + i) % len(freq_distribution)]
                    if freq_distribution else "medium")
                for i, word in enumerate(wave)
            ]
            for result in await asyncio.gather(*tasks):
                if result is not None and len(questions) < count:
                    questions.append(result)
            log(f"[INFO] Готово {len(questions)}/{count}")

        if len(questions) < max(5, count // 2):
            raise ValueError(
                f"Критически мало вопросов: {len(questions)}/{count}. "
                f"Словарь класса {word_class} содержит слишком много артефактов "
                f"или специальных терминов."
            )
        return questions

    def has_word_list(self, word_class: int) -> bool:
        return word_class in self.word_manager.relative_lists and len(self.word_manager.relative_lists[word_class]) > 0

    def get_available_classes(self) -> list[int]:
        return sorted(self.word_manager.relative_lists.keys())
