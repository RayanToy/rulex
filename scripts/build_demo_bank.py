#!/usr/bin/env python
"""Банк вопросов для демо — генерация тем же пайплайном, что и в приложении.

Результат, data/demo_questions.json, загружается в пустую базу при старте
(app/core/seed.py). Поэтому публичный экземпляр работает без ключа модели:
посетители не тратят деньги на вызовы модели и не могут их потратить.

Классы сохраняются по одному: прерванный прогон продолжается с того места,
где остановился, готовые классы пропускаются (--force генерирует заново).

    python scripts/build_demo_bank.py --classes 4 5 6 7 8 --per-class 25

Классы здесь — классы списков слов: ученику 6 класса соответствует
список 5 класса (см. /api/available-classes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.generation import QuestionGenerator  # noqa: E402
from app.services.generation.console import log  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "demo_questions.json"

ABOUT = ("Банк вопросов для демо. Сгенерирован scripts/build_demo_bank.py тем же "
         "пайплайном, что и в приложении; загружается в пустую базу при старте.")


def record(question: dict) -> dict:
    """Вопрос из генератора в виде записи банка."""
    return {
        "target_word": question["target_word"],
        "definition": question["definition"],
        "correct_answer": question["correct_answer"],
        "distractors": question["distractors"],
        "part_of_speech": question["part_of_speech"],
        "word_class": question["word_class"],
        "frequency_type": question["frequency_type"],
        "generation_log": json.loads(question["generation_log"]),
    }


def load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"about": ABOUT, "classes": {}, "questions": []}


def save(path: Path, bank: dict) -> None:
    bank["questions"].sort(key=lambda q: (q["word_class"], q["target_word"]))
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", type=int, nargs="+", default=[4, 5, 6, 7, 8],
                        help="классы списков слов (ученику N класса — список N-1)")
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="сгенерировать готовые классы заново")
    args = parser.parse_args()

    gen = QuestionGenerator()
    bank = load(args.out)
    log(f"[INFO] модель {gen.model}, классы {args.classes}, по {args.per_class} вопросов → {args.out}")

    for word_class in args.classes:
        if str(word_class) in bank["classes"] and not args.force:
            log(f"[SKIP] класс {word_class}: уже в банке")
            continue

        started = time.perf_counter()
        questions = gen.generate_questions_for_class(word_class, count=args.per_class)
        bank["questions"] = [q for q in bank["questions"] if q["word_class"] != word_class]
        bank["questions"] += [record(q) for q in questions]
        bank["classes"][str(word_class)] = {
            "questions": len(questions),
            "model": gen.model,
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        save(args.out, bank)
        log(f"[OK] класс {word_class}: {len(questions)} вопросов за {time.perf_counter() - started:.0f} с")

    s = gen.structured_stats
    log(f"[INFO] всего в банке {len(bank['questions'])}; структурных ответов в прогоне: "
        f"{s['tool']}, разобрано из текста: {s['fallback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
