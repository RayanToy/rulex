#!/usr/bin/env python
"""Прогон eval для пайплайна генерации вопросов RuLex.

Измеряет две разные вещи:

  1. Качество ФИЛЬТРА — отличает ли пайплайн реальные слова от мусора корпуса.
     Это основная ценность проекта: списки class_N_relative.csv зашумлены.
  2. Качество ГЕНЕРАЦИИ — на словах, размеченных как реальные: утечка ответа
     в толкование, совпадение части речи у дистракторов, доля успеха.

Плюс стоимость и латентность по каждому вызову API.

Запуск:
    python eval/run_eval.py --dry-run     # только бесплатные эвристики
    python eval/run_eval.py               # полный прогон (тратит деньги)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import metrics  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"
CACHE_PATH = ROOT / "eval" / ".cache.json"


def safe_name(value: str) -> str:
    """Имя, пригодное для файловой системы.

    Идентификаторы моделей содержат двоеточие (`qwen2.5:7b-instruct`),
    а Windows трактует его как разделитель альтернативного потока NTFS:
    файл не падает с ошибкой, а молча пишется в скрытый поток и потом
    не находится. Точка тоже заменяется — ради единообразия имён.
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")


# --------------------------- перехват вызовов API ---------------------------

class LLMRecorder:
    """Оборачивает client.messages.create: пишет usage/латентность и кэширует ответы.

    Кэш нужен, чтобы повторный прогон на тех же промптах не стоил денег.
    Ключ — хэш от модели, system, messages, max_tokens, tools и output_config.
    Хранятся блоки ответа целиком: текст и вызовы инструментов. Блоки
    рассуждения не хранятся — на повторе они не нужны, а текстовый блок
    ищется по типу, поэтому их отсутствие ничего не ломает.
    """

    def __init__(self, client, use_cache: bool = True):
        self._create = client.messages.create
        self.calls: list[dict] = []
        self.use_cache = use_cache
        self.cache: dict = {}
        if use_cache and CACHE_PATH.exists():
            try:
                self.cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}
        client.messages.create = self  # подменяем

    @staticmethod
    def _key(kwargs) -> str:
        # tools и output_config входят в ключ: иначе структурный и обычный
        # запрос с одинаковым промптом делили бы одну запись кэша.
        payload = json.dumps(
            {
                "model": kwargs.get("model"),
                "system": kwargs.get("system"),
                "messages": kwargs.get("messages"),
                "max_tokens": kwargs.get("max_tokens"),
                "tools": kwargs.get("tools"),
                "output_config": kwargs.get("output_config"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _dump_blocks(content) -> list[dict]:
        """Блоки ответа в виде, пригодном для кэша: текст и вызовы инструментов.

        Раньше кэшировался только текст, и на повторном прогоне
        структурный ответ возвращался текстом — харнесс показал бы 0%
        структурных ответов, то есть измерял бы сам себя.
        """
        blocks = []
        for block in content or []:
            kind = getattr(block, "type", None)
            if kind == "text" and hasattr(block, "text"):
                blocks.append({"type": "text", "text": block.text})
            elif kind == "tool_use":
                blocks.append({"type": "tool_use", "name": block.name,
                               "input": dict(block.input)})
        return blocks

    @staticmethod
    def _load_blocks(hit: dict) -> list:
        if "blocks" in hit:
            return [
                SimpleNamespace(type="tool_use", id="cached", name=b["name"], input=b["input"])
                if b["type"] == "tool_use" else SimpleNamespace(type="text", text=b["text"])
                for b in hit["blocks"]
            ]
        # записи кэша старого формата
        return [SimpleNamespace(type="text", text=hit.get("text", ""))]

    def __call__(self, **kwargs):
        key = self._key(kwargs)
        model = kwargs.get("model", "?")

        if self.use_cache and key in self.cache:
            hit = self.cache[key]
            self.calls.append({
                "model": model,
                "cached": True,
                "latency_s": 0.0,
                "input_tokens": hit["input_tokens"],
                "output_tokens": hit["output_tokens"],
                "cost_usd": metrics.cost_usd(model, hit["input_tokens"], hit["output_tokens"]),
            })
            return SimpleNamespace(
                stop_reason="end_turn",
                # type обязателен: потребитель отбирает блоки по типу
                content=self._load_blocks(hit),
                usage=SimpleNamespace(
                    input_tokens=hit["input_tokens"],
                    output_tokens=hit["output_tokens"],
                ),
            )

        started = time.perf_counter()
        response = self._create(**kwargs)
        latency = time.perf_counter() - started

        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0

        self.calls.append({
            "model": model,
            "cached": False,
            "latency_s": round(latency, 3),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cost_usd": metrics.cost_usd(model, in_tok, out_tok),
        })

        if self.use_cache:
            self.cache[key] = {
                "blocks": self._dump_blocks(response.content),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
        return response

    def flush(self):
        if self.use_cache:
            CACHE_PATH.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")

    def summary(self) -> dict:
        live = [c for c in self.calls if not c["cached"]]
        lat = [c["latency_s"] for c in live]
        # Совместимые шлюзы часто не проставляют usage. Если ответ пришёл,
        # а входных токенов ноль — учёт стоимости недостоверен, и об этом
        # нельзя молчать: цифры из такого прогона нельзя публиковать.
        no_usage = sum(1 for c in live if c["input_tokens"] == 0 and c["output_tokens"] > 0)
        return {
            "calls_total": len(self.calls),
            "calls_live": len(live),
            "calls_cached": len(self.calls) - len(live),
            "calls_without_usage": no_usage,
            "usage_reliable": no_usage == 0,
            "input_tokens": sum(c["input_tokens"] for c in self.calls),
            "output_tokens": sum(c["output_tokens"] for c in self.calls),
            "cost_usd": round(sum(c["cost_usd"] for c in self.calls), 4),
            "latency_p50_s": round(metrics.percentile(lat, 50), 3),
            "latency_p95_s": round(metrics.percentile(lat, 95), 3),
            "latency_mean_s": round(statistics.mean(lat), 3) if lat else 0.0,
            "models": sorted({c["model"] for c in self.calls}),
        }


# ------------------------------ загрузка данных -----------------------------

def load_words(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rows.append({
            "word_class": int(parts[0]),
            "word": parts[1].strip(),
            "label": parts[2].strip(),
            "note": parts[3].strip() if len(parts) > 3 else "",
        })
    return rows


# ------------------------------- этапы прогона ------------------------------

def eval_filter(gen, rows: list[dict], use_llm: bool) -> dict:
    """Насколько точно пайплайн отсеивает мусор корпуса."""
    words = [r["word"] for r in rows]
    truth = [r["label"] for r in rows]

    # Этап 1 — бесплатные эвристики (_is_basic_valid + _is_artifact)
    heur_pred, heur_reason = [], {}
    for w in words:
        ok_basic, why_basic = gen._is_basic_valid(w)
        if not ok_basic:
            heur_pred.append("artifact")
            heur_reason[w] = why_basic
            continue
        is_art, why_art = gen._is_artifact(w)
        heur_pred.append("artifact" if is_art else "real")
        if is_art:
            heur_reason[w] = why_art

    result = {
        "heuristic": metrics.confusion(truth, heur_pred),
        "heuristic_rejections": heur_reason,
        "per_word": [
            {"word": w, "truth": t, "heuristic": h}
            for w, t, h in zip(words, truth, heur_pred, strict=True)
        ],
    }

    if not use_llm:
        return result

    # Этап 2 — батч-проверка через LLM поверх выживших после эвристик
    survivors = [w for w, p in zip(words, heur_pred, strict=True) if p == "real"]
    kept = set(gen._filter_real_words_batch(survivors, batch_size=30))

    combined = [
        "artifact" if (p == "artifact" or w not in kept) else "real"
        for w, p in zip(words, heur_pred, strict=True)
    ]
    result["llm_kept"] = sorted(kept)
    result["combined"] = metrics.confusion(truth, combined)
    for item, c in zip(result["per_word"], combined, strict=True):
        item["combined"] = c
    return result


def eval_generation(gen, rows: list[dict]) -> dict:
    """Качество генерации на словах, размеченных как реальные."""
    targets = [r for r in rows if r["label"] == "real"]
    items, failures = [], []

    for r in targets:
        word, cls = r["word"], r["word_class"]
        try:
            q = gen.generate_question(word, cls)
        except Exception as exc:  # слово отбраковано пайплайном или сбой API
            failures.append({"word": word, "error": f"{type(exc).__name__}: {exc}"})
            continue

        distractors = q.get("distractors", []) or []
        leaks = metrics.leaked_words(q.get("definition", ""), [word, *distractors])
        foreign = metrics.foreign_letters(q.get("definition", "") + " " + " ".join(distractors))
        dupes = metrics.duplicate_distractors(distractors)
        items.append({
            "foreign_letters": foreign,
            "duplicate_distractors": dupes,
            "word": word,
            "definition": q.get("definition", ""),
            "distractors": distractors,
            "n_distractors": len(distractors),
            "leaked_words": leaks,
            "has_leak": bool(leaks),
            "pos_match_ratio": metrics.pos_match_ratio(word, distractors),
        })

    attempted = len(targets)
    pos_ratios = [i["pos_match_ratio"] for i in items if i["pos_match_ratio"] is not None]
    return {
        "attempted": attempted,
        "succeeded": len(items),
        "yield": round(len(items) / attempted, 4) if attempted else 0.0,
        "definition_leak_rate": (
            round(sum(i["has_leak"] for i in items) / len(items), 4) if items else None
        ),
        "pos_match_mean": round(statistics.mean(pos_ratios), 4) if pos_ratios else None,
        "foreign_script_rate": (
            round(sum(1 for i in items if i["foreign_letters"]) / len(items), 4) if items else None
        ),
        "duplicate_distractor_rate": (
            round(sum(1 for i in items if i["duplicate_distractors"]) / len(items), 4) if items else None
        ),
        "distractors_mean": (
            round(statistics.mean([i["n_distractors"] for i in items]), 2) if items else None
        ),
        "items": items,
        "failures": failures,
    }


# -------------------------------- отчётность --------------------------------

def to_markdown(report: dict) -> str:
    lines = []
    lines.append("# RuLex — отчёт eval\n")
    lines.append("- прогон: `{}`".format(report["run_id"]))
    lines.append("- модель генерации: `{}`".format(report["model"]))
    lines.append("- набор: `{}` ({} слов)".format(report["words_file"], report["n_words"]))
    lines.append("- режим: {}\n".format("сухой (без API)" if report["dry_run"] else "полный"))

    flt = report.get("filter") or {}
    if flt:
        lines.append("## Фильтрация мусора корпуса\n")
        lines.append("| этап | precision | recall | F1 | accuracy |")
        lines.append("|---|---|---|---|---|")
        h = flt["heuristic"]
        lines.append("| эвристики (без LLM) | {} | {} | {} | {} |".format(
            h["precision"], h["recall"], h["f1"], h["accuracy"]))
        if "combined" in flt:
            c = flt["combined"]
            lines.append("| эвристики + LLM | {} | {} | {} | {} |".format(
                c["precision"], c["recall"], c["f1"], c["accuracy"]))
        lines.append("")
        lines.append(
            "Положительный класс — `artifact`; recall показывает, "
            "какую долю мусора удалось поймать.\n"
        )

    gen = report.get("generation") or {}
    if gen:
        lines.append("## Качество генерации\n")
        lines.append("| метрика | значение |")
        lines.append("|---|---|")
        lines.append("| доля успеха (yield) | {} ({}/{}) |".format(
            gen["yield"], gen["succeeded"], gen["attempted"]))
        lines.append("| утечка ответа в толкование | {} |".format(gen["definition_leak_rate"]))
        lines.append("| совпадение части речи у дистракторов | {} |".format(gen["pos_match_mean"]))
        lines.append("| нерусские буквы в задании | {} |".format(gen.get("foreign_script_rate")))
        lines.append("| повторяющиеся дистракторы | {} |".format(gen.get("duplicate_distractor_rate")))
        lines.append("| дистракторов на вопрос (среднее) | {} |".format(gen["distractors_mean"]))
        lines.append("")

    structured = report.get("structured") or {}
    if structured.get("share") is not None:
        lines.append("## Структурированные ответы\n")
        lines.append("| ответ | вызовов |")
        lines.append("|---|---|")
        lines.append("| по схеме (tool use) | {} |".format(structured.get("tool", 0)))
        lines.append("| текстом, разобран запаской | {} |".format(structured.get("fallback", 0)))
        lines.append("| доля структурных | {:.0%} |".format(structured["share"]))
        lines.append("")

    usage = report.get("usage") or {}
    if usage:
        lines.append("## Стоимость и латентность\n")
        lines.append("| метрика | значение |")
        lines.append("|---|---|")
        lines.append("| вызовов API (живых / из кэша) | {} / {} |".format(
            usage["calls_live"], usage["calls_cached"]))
        lines.append("| токенов вход / выход | {} / {} |".format(
            usage["input_tokens"], usage["output_tokens"]))
        lines.append("| стоимость прогона | ${} |".format(usage["cost_usd"]))
        if report.get("cost_per_question") is not None:
            lines.append("| стоимость вопроса | ${} |".format(report["cost_per_question"]))
        lines.append("| латентность p50 / p95 | {} с / {} с |".format(
            usage["latency_p50_s"], usage["latency_p95_s"]))
        lines.append("")
        if not usage.get("usage_reliable", True):
            lines.append(
                "> **Стоимость недостоверна.** {} из {} живых вызовов вернулись "
                "с `input_tokens = 0` — шлюз не проставляет usage. "
                "Эти цифры нельзя публиковать как измеренные; для честного "
                "замера нужен прямой ключ Anthropic.\n".format(
                    usage["calls_without_usage"], usage["calls_live"])
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--words", type=Path, default=ROOT / "eval" / "words.tsv")
    parser.add_argument("--dry-run", action="store_true",
                        help="только бесплатные эвристики, без вызовов API")
    parser.add_argument("--no-cache", action="store_true", help="игнорировать кэш ответов")
    parser.add_argument("--limit", type=int, default=0, help="взять только первые N слов")
    parser.add_argument("--label", default="", help="пометка прогона (например, baseline)")
    args = parser.parse_args()

    rows = load_words(args.words)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("Набор пуст — нечего прогонять", file=sys.stderr)
        return 1

    if args.dry_run:
        os.environ.setdefault("ANTHROPIC_API_KEY", "dry-run-placeholder")

    from app.services.generator import QuestionGenerator

    gen = QuestionGenerator()
    recorder = None if args.dry_run else LLMRecorder(gen.client, use_cache=not args.no_cache)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.label:
        run_id = f"{run_id}-{safe_name(args.label)}"

    report = {
        "run_id": run_id,
        "dry_run": args.dry_run,
        "model": gen.model,
        "words_file": str(args.words.relative_to(ROOT)),
        "n_words": len(rows),
        "labels": {
            "real": sum(r["label"] == "real" for r in rows),
            "artifact": sum(r["label"] == "artifact" for r in rows),
        },
    }

    report["filter"] = eval_filter(gen, rows, use_llm=not args.dry_run)
    if not args.dry_run:
        report["generation"] = eval_generation(gen, rows)
        report["usage"] = recorder.summary()
        recorder.flush()
        stats = dict(getattr(gen, "structured_stats", {}) or {})
        total = sum(stats.values())
        report["structured"] = {
            **stats,
            "share": round(stats.get("tool", 0) / total, 4) if total else None,
        }
        done = report["generation"]["succeeded"]
        report["cost_per_question"] = (
            round(report["usage"]["cost_usd"] / done, 4) if done else None
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{run_id}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = to_markdown(report)
    (RESULTS_DIR / f"{run_id}.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nОтчёты: eval/results/{run_id}.json, eval/results/{run_id}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
