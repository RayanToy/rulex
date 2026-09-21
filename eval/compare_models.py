#!/usr/bin/env python
"""Сравнение моделей на задаче RuLex.

Прогоняет один и тот же набор слов через один и тот же пайплайн, меняя
ровно модель. Всё остальное — промпты, эвристики, набор, метрики —
одинаково, иначе сравнивались бы не модели, а условия.

    python eval/compare_models.py
    python eval/compare_models.py --models claude-haiku-4-5 claude-sonnet-5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from run_eval import safe_name  # noqa: E402

DEFAULT_MODELS = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]


def run_one(model: str, words: Path, extra_env: dict) -> dict | None:
    env = {**os.environ, **extra_env, "RULEX_MODEL_GENERATION": model}
    label = safe_name("model-" + model)
    cmd = [
        sys.executable, str(ROOT / "eval" / "run_eval.py"),
        "--words", str(words), "--label", label, "--no-cache",
    ]
    print(f"\n{'=' * 70}\n  {model}\n{'=' * 70}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"  ОШИБКА (код {result.returncode}):", flush=True)
        print("  " + "\n  ".join(result.stderr.strip().splitlines()[-8:]), flush=True)
        return None

    reports = sorted((ROOT / "eval" / "results").glob(f"*{label}.json"))
    if not reports:
        print("  отчёт не найден", flush=True)
        return None
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def summarize(reports: dict[str, dict]) -> str:
    lines = ["", "=" * 78, "СРАВНЕНИЕ МОДЕЛЕЙ НА ЗАДАЧЕ RuLex", "=" * 78, ""]
    header = f"{'модель':<22}{'yield':>8}{'F1 фильтр':>12}{'recall':>9}{'утечка':>9}{'p50 с':>8}{'p95 с':>8}"
    lines += [header, "-" * len(header)]
    for model, rep in reports.items():
        if not rep:
            lines.append(f"{model:<22}{'— прогон не удался':>30}")
            continue
        gen, flt, use = rep["generation"], rep["filter"]["combined"], rep["usage"]
        lines.append(
            f"{model:<22}{gen['yield']:>8.3f}{flt['f1']:>12.3f}{flt['recall']:>9.2f}"
            f"{(gen['definition_leak_rate'] if gen['definition_leak_rate'] is not None else -1):>9.2f}"
            f"{use['latency_p50_s']:>8.1f}{use['latency_p95_s']:>8.1f}"
        )
    lines += ["", "yield — доля слов, для которых задание построено до конца",
              "F1/recall — отсев мусора корпуса, положительный класс artifact",
              "утечка — доля толкований, где ответ виден в формулировке", ""]

    unreliable = [m for m, r in reports.items()
                  if r and not r["usage"].get("usage_reliable", True)]
    if unreliable:
        lines += [
            "ВНИМАНИЕ: стоимость не измерена — шлюз вернул input_tokens = 0 "
            f"для: {', '.join(unreliable)}.",
            "Сравнивать модели по цене на этих данных нельзя.", "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--words", type=Path, default=ROOT / "eval" / "words.tsv")
    parser.add_argument("--base-url", default=os.getenv("ANTHROPIC_BASE_URL", ""))
    args = parser.parse_args()

    extra_env = {"ANTHROPIC_BASE_URL": args.base_url} if args.base_url else {}
    reports = {model: run_one(model, args.words, extra_env) for model in args.models}

    text = summarize(reports)
    print(text)
    out = ROOT / "eval" / "results" / "model-comparison.md"
    out.write_text(text, encoding="utf-8")
    print(f"Сводка: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
