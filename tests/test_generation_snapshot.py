"""Снимок поведения генератора: запросы к модели и результат.

От запросов к модели зависят и качество заданий, и ключи кэша
eval-харнесса. Снимок фиксирует их байт в байт вместе с результатом
генерации, журналом generation_log и порядком вызовов. Случайная правка
промпта при рефакторинге видна как падение теста, намеренная — как дифф
golden-файла в ревью.

Модель — сценарий по правилам: какие слова «выдуманы», какие не подходят
для теста, какое толкование вернуть. Сеть не используется.

Обновить снимок после намеренного изменения промпта или логики:
    RULEX_UPDATE_GOLDEN=1 pytest tests/test_generation_snapshot.py
"""
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.generation import model_calls, pipeline, realness  # noqa: E402
from app.services.llm import OllamaError  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "generation.json"

FAKE_WORDS = {"жоут", "травие", "ведрик"}   # модель считает выдуманными
UNSUITABLE = {"москва"}                     # модель считает непригодными для теста
NOUN_DISTRACTORS = ["племя", "народ", "житель", "предок", "камень"]
VERB_DISTRACTORS = ["бежать", "идти", "прыгать", "ползти", "лететь"]
DEFAULT_DEFINITION = "Понятие, знакомое каждому школьнику"

# Частота по словарю Шарова, ipm; остальных слов в словаре «нет»
SHAROV = {"книга": 250.0, "город": 300.0, "окно": 120.0, "школа": 200.0, "читать": 150.0,
          "река": 80.0, "гора": 70.0, "берег": 40.0, "мороз": 30.0, "сад": 50.0,
          "лампа": 20.0, "бежать": 60.0}

CLASS_WORDS = [
    # реальные слова, которые подтверждают словари
    "книга", "река", "город", "лампа", "мороз", "берег", "окно", "школа",
    "гора", "сад", "бежать", "читать", "строить", "москва",
    # неизвестны словарям — решает модель
    "травие", "ведрик", "восьмибрат",
    # мусор, который отсекают эвристики
    "пкно", "тпт", "ab",
]


def tool_use(name, payload):
    return SimpleNamespace(type="tool_use", id="t1", name=name, input=payload)


def text(value):
    return SimpleNamespace(type="text", text=value)


class ScriptedModel:
    """Фальшивая модель: отвечает по правилам и записывает запросы.

    Решатель (этап однозначности) видит только толкование и варианты,
    поэтому «правильный ответ» модель берёт из предыдущего запроса о том
    же слове. ambiguous — дистракторы, которые решатель тоже сочтёт
    подходящими; solver_misses — решатель не узнаёт загаданное слово.
    """

    def __init__(self, definitions=(), fail=False, ambiguous=(), solver_misses=False):
        self.requests = []
        self.messages = self
        self._definitions = list(definitions)
        self._fail = fail
        self._ambiguous = set(ambiguous)
        self._solver_misses = solver_misses
        self._last_word = None

    def create(self, **params):
        self.requests.append(params)
        if self._fail:
            raise OllamaError("нет связи")
        return SimpleNamespace(content=self._answer(params), stop_reason="end_turn")

    def _answer(self, params):
        prompt = params["messages"][0]["content"]
        tool = (params.get("tools") or [{}])[0].get("name")
        if tool == "report_matching_options":
            options = listed_options(prompt)
            chosen = [o for o in options if o != self._last_word] if self._solver_misses else [self._last_word]
            matching = chosen[:1] + [o for o in options if o in self._ambiguous]
            return [tool_use(tool, {"matching": matching, "reason": "по толкованию"})]
        if re.search(r'слов[а-я]* "', prompt):
            self._last_word = quoted_word(prompt)
        if tool == "report_real_words":
            listed = listed_words(prompt)
            return [tool_use(tool, {"real_words": [w for w in listed if w not in FAKE_WORDS]})]
        if tool == "report_suitability":
            ok = quoted_word(prompt) not in UNSUITABLE
            return [tool_use(tool, {"suitable": ok, "reason": "общеупотребительное" if ok else "топоним"})]
        if tool == "report_distractors":
            verb = "(глагол)" in prompt or "(инфинитив)" in prompt
            return [tool_use(tool, {"distractors": VERB_DISTRACTORS if verb else NOUN_DISTRACTORS})]
        if self._definitions:
            return [text(self._definitions.pop(0))]
        return [text(DEFAULT_DEFINITION)]


def listed_words(prompt):
    return prompt.split("Список для проверки:\n", 1)[1].split("\n", 1)[0].split(", ")


def quoted_word(prompt):
    return re.search(r'слов[а-я]* "([^"]+)"', prompt).group(1)


def listed_options(prompt):
    return prompt.split("Варианты ответа: ", 1)[1].split("\n", 1)[0].split(", ")


def request_summary(params):
    """Короткая запись запроса для сценариев уровня класса."""
    prompt = params["messages"][0]["content"]
    tool = (params.get("tools") or [{}])[0].get("name")
    if tool == "report_real_words":
        return f"{tool}: {', '.join(listed_words(prompt))}"
    if tool == "report_matching_options":
        return f"{tool}: {', '.join(listed_options(prompt))}"
    if tool:
        return f"{tool}: {quoted_word(prompt)}"
    kind = "correction" if prompt.startswith("Исправь толкование") else "definition"
    return f"{kind}: {prompt.splitlines()[0][:60]}"


class FakeWordManager:
    def __init__(self, lists):
        self.relative_lists = lists

    def get_words_for_class(self, word_class):
        return list(self.relative_lists.get(word_class, []))

    def get_sharov_frequency(self, word):
        return SHAROV.get(word, 0)


def make_generator(mp, model, lists=None):
    """Генератор с фальшивой моделью и словарями."""
    mp.setattr(realness, "get_verdicts", lambda: {})
    llm = model_calls.ModelCaller(client=model, model="test-model")
    return pipeline.QuestionGenerator(llm=llm, word_manager=FakeWordManager(lists or {}))


def question_view(question):
    view = {k: v for k, v in question.items() if k != "generation_log"}
    view["generation_log"] = json.loads(question["generation_log"])
    return view


def error_view(exc):
    return f"{type(exc).__name__}: {exc}"


# --------------------------------- сценарии ---------------------------------

def scenario_question_structured(mp):
    model = ScriptedModel(definitions=["Древний человек, живший в эпоху палеолита"])
    gen = make_generator(mp, model)
    result = gen.generate_question("Кроманьонец", 6, "low")
    return {"requests": model.requests, "result": question_view(result)}


def scenario_question_with_correction(mp):
    # «жителей» — однокоренное с дистрактором «житель», нужна правка
    model = ScriptedModel(definitions=["много деревьев и жителей вокруг.",
                                       "- Большая территория, где растут деревья"])
    gen = make_generator(mp, model)
    result = gen.generate_question("лес", 5, "medium")
    return {"requests": model.requests, "result": question_view(result)}


def scenario_question_ambiguous_distractor_dropped(mp):
    """Решатель счёл «народ» подходящим под толкование — вариант убирается."""
    model = ScriptedModel(definitions=["Древний человек, живший в эпоху палеолита"],
                          ambiguous={"народ"})
    gen = make_generator(mp, model)
    result = gen.generate_question("кроманьонец", 6, "low")
    return {"requests": [request_summary(p) for p in model.requests], "result": question_view(result)}


def scenario_question_ambiguous_rejected(mp):
    """Подходят два дистрактора из трёх — задание отбраковывается."""
    model = ScriptedModel(definitions=["Древний человек, живший в эпоху палеолита"],
                          ambiguous={"народ", "племя"})
    gen = make_generator(mp, model)
    with pytest.raises(ValueError) as exc:
        gen.generate_question("кроманьонец")
    return {"error": error_view(exc.value), "generation_log": gen.generation_log}


def scenario_question_solver_misses(mp):
    """Решатель не узнал загаданное слово — толкование на него не указывает."""
    model = ScriptedModel(definitions=["Древний человек, живший в эпоху палеолита"],
                          solver_misses=True)
    gen = make_generator(mp, model)
    with pytest.raises(ValueError) as exc:
        gen.generate_question("кроманьонец")
    return {"error": error_view(exc.value)}


def scenario_question_leak_rejected(mp):
    model = ScriptedModel(definitions=["Мебель: стол для еды"] * 4)
    gen = make_generator(mp, model)
    with pytest.raises(ValueError) as exc:
        gen.generate_question("стол")
    return {"requests": model.requests, "error": error_view(exc.value),
            "generation_log": gen.generation_log}


def scenario_question_bad_word(mp):
    model = ScriptedModel()
    gen = make_generator(mp, model)
    with pytest.raises(ValueError) as exc:
        gen.generate_question("красивый")
    return {"requests": model.requests, "error": error_view(exc.value)}


def scenario_filter_batches(mp):
    model = ScriptedModel()
    gen = make_generator(mp, model)
    kept = gen.realness.filter_batch(["жоут", "стол", "книга", "травие", "река"], batch_size=2)
    return {"requests": model.requests, "kept": kept, "failures": gen.realness.failures,
            "structured_stats": gen.structured_stats}


def scenario_filter_api_error(mp):
    model = ScriptedModel(fail=True)
    gen = make_generator(mp, model)
    kept = gen.realness.filter_batch(["стол", "книга"])
    return {"kept": kept, "failures": gen.realness.failures}


def scenario_suitability(mp):
    model = ScriptedModel()
    gen = make_generator(mp, model)
    verdicts = [gen.suitability.check("стол"), gen.suitability.check("москва")]
    return {"requests": model.requests, "verdicts": verdicts, "generation_log": gen.generation_log}


def scenario_suitability_api_error(mp):
    gen = make_generator(mp, ScriptedModel(fail=True))
    return {"verdict": gen.suitability.check("стол"), "generation_log": gen.generation_log}


def scenario_prefilter_interface(mp):
    """То, чем пользуется scripts/prefilter_corpus.py: параметры запроса
    и разбор ответа отдельно от вызова."""
    gen = make_generator(mp, ScriptedModel())
    batch = ["стол", "жоут", "книга"]
    answer = [tool_use("report_real_words", {"real_words": ["**Стол**", "книга", "лишнее"]})]
    return {
        "params": gen.realness.request_params(batch),
        "confirmed": gen.realness.words_from_answer(batch, answer),
        "heuristics": {w: [realness.is_basic_valid(w), realness.is_artifact(w)]
                       for w in ["стол", "пкно", "ab", "красивый", "бежать"]},
        "dictionary": {w: gen.realness.is_dictionary_word(w) for w in ["стол", "восьмибрат"]},
    }


def scenario_class_sync(mp):
    model = ScriptedModel()
    gen = make_generator(mp, model, {6: CLASS_WORDS})
    random.seed(20260925)
    questions = gen.generate_questions_for_class(6, count=5)
    return {
        "has_word_list": [gen.has_word_list(6), gen.has_word_list(7)],
        "available_classes": gen.get_available_classes(),
        "questions": [question_view(q) for q in questions],
        "requests": [request_summary(p) for p in model.requests],
    }


def scenario_class_async(mp):
    model = ScriptedModel()
    gen = make_generator(mp, model, {6: CLASS_WORDS})
    random.seed(20260925)
    questions = asyncio.run(gen.agenerate_questions_for_class(6, count=5, concurrency=1))
    return {
        "questions": [question_view(q) for q in questions],
        "requests": [request_summary(p) for p in model.requests],
    }


def scenario_class_too_few_words(mp):
    gen = make_generator(mp, ScriptedModel(), {6: ["книга", "пкно"]})
    with pytest.raises(ValueError) as exc:
        gen.generate_questions_for_class(6, count=5)
    return {"error": error_view(exc.value)}


SCENARIOS = {name[len("scenario_"):]: fn for name, fn in globals().items()
             if name.startswith("scenario_")}


def normalized(value):
    """Кортежи -> списки и т. п.: сравнивается то, что лежит в JSON."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


@pytest.fixture(scope="module")
def actual():
    results = {}
    for name, scenario in SCENARIOS.items():
        with pytest.MonkeyPatch.context() as mp:
            results[name] = normalized(scenario(mp))
    return results


UPDATE = bool(os.getenv("RULEX_UPDATE_GOLDEN"))


@pytest.mark.skipif(UPDATE, reason="режим обновления снимка")
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_matches_golden(actual, name):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert name in golden, f"сценария {name} нет в снимке — обновите его"
    assert actual[name] == golden[name]


@pytest.mark.skipif(not UPDATE, reason="снимок обновляется только с RULEX_UPDATE_GOLDEN=1")
def test_update_golden(actual):
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(actual, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
