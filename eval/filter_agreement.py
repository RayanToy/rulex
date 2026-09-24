#!/usr/bin/env python
"""Согласие бесплатного словарного фильтра и платного LLM-фильтра.

Отвечает на вопрос «нужен ли вообще платный шаг проверки реальности».
Вместо ручной разметки сотен слов меряем, насколько два метода расходятся:
размечать глазами нужно только разногласия, и именно они решают дело.

Набор `filter_agreement_words.tsv` стратифицирован 50/50 по словарной
проверке — иначе он тривиально решался бы ею же и ничего не показал.

    python eval/filter_agreement.py
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.services import generator  # noqa: E402

WORDS_FILE = ROOT / "eval" / "filter_agreement_words.tsv"
RESULTS = ROOT / "eval" / "results"


def load_words() -> list[tuple[str, bool]]:
    rows = []
    for line in WORDS_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        word, known = line.split("\t")
        rows.append((word, known == "1"))
    return rows


def main() -> int:
    # Аргументов нет, но --help не должен запускать платный прогон
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    words = load_words()
    out = io.StringIO()

    gen = generator.QuestionGenerator()
    print(f"модель: {gen.model}   слов: {len(words)}", file=out)

    kept = set(gen._filter_real_words_batch([w for w, _ in words], batch_size=30))
    if gen.filter_failures:
        # Слова из упавшего батча выглядят как «LLM отверг» — такие цифры
        # нельзя писать поверх настоящих результатов
        print(f"[ERROR] {len(gen.filter_failures)} батчей не выполнено из-за сбоя API, "
              f"результаты не записаны", file=sys.stderr)
        return 1

    rows = [(w, known, w in kept) for w, known in words]
    agree = [r for r in rows if r[1] == r[2]]
    disagree = [r for r in rows if r[1] != r[2]]

    print(f"\nсогласие: {len(agree)}/{len(rows)} = {len(agree) / len(rows):.1%}", file=out)
    print(f"разногласий: {len(disagree)}\n", file=out)

    only_llm = [r[0] for r in disagree if r[2] and not r[1]]
    only_dict = [r[0] for r in disagree if r[1] and not r[2]]
    print(f"--- LLM считает реальным, словарь не знает ({len(only_llm)}) ---", file=out)
    print(", ".join(only_llm), file=out)
    print(f"\n--- словарь знает, LLM отверг ({len(only_dict)}) ---", file=out)
    print(", ".join(only_dict), file=out)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "filter-disagreements.txt").write_text(
        "\n".join(["# LLM=real, словарь не знает", *only_llm,
                   "", "# словарь знает, LLM отверг", *only_dict]),
        encoding="utf-8")
    (RESULTS / "filter-agreement.txt").write_text(out.getvalue(), encoding="utf-8")
    print(out.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
