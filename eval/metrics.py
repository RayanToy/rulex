"""Чистые функции метрик. Без сети и без обращений к LLM — тестируются offline."""
from __future__ import annotations

import functools
from typing import Iterable, Sequence

import pymorphy3

_morph = pymorphy3.MorphAnalyzer()

# Цены API, $ за 1M токенов. Источник — прайс Anthropic.
# claude-sonnet-4-20250514 — снятая с поддержки модель, на которой снимался baseline;
# её ставка указана по историческому прайсу и помечена как приблизительная.
PRICING = {
    "claude-opus-5":            (5.00, 25.00),
    "claude-sonnet-5":          (2.00, 10.00),
    "claude-haiku-4-5":         (1.00,  5.00),
    "claude-sonnet-4-6":        (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),   # legacy, приблизительно
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Стоимость одного вызова. Неизвестная модель -> 0.0, но это видно в отчёте."""
    rate_in, rate_out = PRICING.get(model, (0.0, 0.0))
    return input_tokens / 1_000_000 * rate_in + output_tokens / 1_000_000 * rate_out


@functools.lru_cache(maxsize=100_000)
def lemma(word: str) -> str:
    parsed = _morph.parse(word)
    return parsed[0].normal_form if parsed else word.lower()


@functools.lru_cache(maxsize=100_000)
def pos_of(word: str) -> str | None:
    parsed = _morph.parse(word)
    if not parsed:
        return None
    pos = parsed[0].tag.POS
    # INFN и VERB — одна часть речи для целей теста
    return "VERB" if pos in {"VERB", "INFN"} else (str(pos) if pos else None)


def leaked_words(definition: str, forbidden: Sequence[str]) -> list[str]:
    """Слова толкования, у которых лемма совпала с леммой запретного слова.

    Намеренно проверяется ТОЛЬКО совпадение лемм: оно не даёт ложных
    срабатываний. Однокоренные с чередованием (книга/книжный) так не ловятся —
    это осознанный размен, спорные случаи отдаются модели-судье.
    """
    import re

    banned = {lemma(w) for w in forbidden if w}
    hits = []
    for token in re.findall(r"[а-яёА-ЯЁ]+", definition):
        if lemma(token) in banned:
            hits.append(token)
    return hits


def pos_match_ratio(target: str, distractors: Iterable[str]) -> float | None:
    """Доля дистракторов, совпавших с целевым словом по части речи."""
    items = [d for d in distractors if d]
    if not items:
        return None
    want = pos_of(target)
    if want is None:
        return None
    return sum(1 for d in items if pos_of(d) == want) / len(items)


def confusion(y_true: Sequence[str], y_pred: Sequence[str], positive: str = "artifact") -> dict:
    """Матрица ошибок. Положительный класс по умолчанию — 'artifact':
    нас интересует, насколько хорошо фильтр ловит мусор."""
    if len(y_true) != len(y_pred):
        raise ValueError("длины y_true и y_pred не совпадают")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p != positive)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def percentile(values: Sequence[float], p: float) -> float:
    """Перцентиль методом ближайшего ранга. Пустой вход -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(p / 100 * len(ordered) + 0.5)) - 1))
    return ordered[idx]
