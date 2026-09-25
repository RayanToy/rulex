#!/usr/bin/env python
"""Офлайн-предфильтрация корпуса: вердикт «реальное слово или мусор» один раз.

Вердикт о слове не меняется от запроса к запросу, а раньше считался заново
при каждой генерации. Здесь он считается один раз для всего корпуса и
сохраняется в data/word_verdicts.tsv; пайплайн берёт его оттуда до словарей
и до модели.

Порядок тот же, что в живом каскаде: эвристики пайплайна, словари
(pymorphy3 + Шаров), модель — только для оставшегося. Промпт и разбор
ответа модели — те же, что в живом пути: этап RealnessFilter.

Модель вызывается через Batches API, если он доступен (вдвое дешевле и без
лимитов на одновременность), иначе — по одному запросу. Шлюз router.cheap
Batches API не поддерживает, и переход происходит автоматически.

Задача возобновляемая: слова с сохранённым вердиктом пропускаются, а по
запросам, которые не выполнились, вердикт не пишется — они уйдут
в следующий запуск.

    python scripts/prefilter_corpus.py --classes 6 --limit 90
    python scripts/prefilter_corpus.py                       # весь корпус
    python scripts/prefilter_corpus.py --no-batches          # только по одному
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services import wordlists  # noqa: E402
from app.services.batches import run_requests  # noqa: E402
from app.services.generation.console import log  # noqa: E402
from app.services.generation.model_calls import ModelCaller  # noqa: E402
from app.services.generation.realness import RealnessFilter, is_artifact, is_basic_valid  # noqa: E402

CHUNK = 30  # столько же слов, сколько в живом батч-фильтре


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(2, 12)))
    parser.add_argument("--limit", type=int, default=0,
                        help="не больше N слов в модель за запуск (0 — без ограничения)")
    parser.add_argument("--no-batches", action="store_true",
                        help="не пытаться использовать Batches API")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    llm = ModelCaller()
    realness = RealnessFilter(llm, wordlists.get_word_manager())
    verdicts = wordlists.get_verdicts()

    # Кандидаты — как в _prepare_candidates: базовая проверка и эвристики
    words: list[str] = []
    seen: set[str] = set()
    for cls in args.classes:
        for w in sorted(realness.word_manager.get_words_for_class(cls)):
            if w in seen:
                continue
            seen.add(w)
            if is_basic_valid(w)[0] and not is_artifact(w)[0]:
                words.append(w)

    pending_dict, pending_llm, already = [], [], 0
    for w in words:
        if w in verdicts:
            already += 1
        elif realness.is_dictionary_word(w):
            pending_dict.append(w)
        else:
            pending_llm.append(w)

    log(f"[INFO] кандидатов {len(words)}: вердикт уже есть у {already}, "
        f"словари решают {len(pending_dict)}, модели нужно {len(pending_llm)}")

    wordlists.append_verdicts([(w, "real", "dict") for w in pending_dict])

    if args.limit:
        pending_llm = pending_llm[: args.limit]
    if not pending_llm:
        log("[INFO] в модель отправлять нечего")
        return 0

    chunks = {
        f"chunk-{n:05d}": pending_llm[i:i + CHUNK]
        for n, i in enumerate(range(0, len(pending_llm), CHUNK))
    }
    requests = {cid: realness.request_params(chunk) for cid, chunk in chunks.items()}
    log(f"[INFO] модель {llm.model}: {len(pending_llm)} слов, {len(requests)} запросов")

    started = time.perf_counter()
    results, mode = run_requests(llm.client, requests, prefer_batches=not args.no_batches,
                                 progress=lambda m: log(f"[INFO] {m}"),
                                 poll_seconds=args.poll_seconds)
    elapsed = time.perf_counter() - started

    rows, skipped = [], 0
    for cid, chunk in chunks.items():
        content = results.get(cid)
        if content is None:
            skipped += len(chunk)  # не выполнен — вердикт не пишем, уйдёт в следующий запуск
            continue
        real = set(realness.words_from_answer(chunk, content))
        source = f"llm:{llm.model}:{mode}"
        rows += [(w, "real" if w in real else "artifact", source) for w in chunk]
    wordlists.append_verdicts(rows)

    n_real = sum(1 for _, v, _ in rows if v == "real")
    log(f"[INFO] режим: {mode}, время: {elapsed:.0f} с")
    log(f"[INFO] записано вердиктов модели: {len(rows)} (реальных {n_real}, мусора {len(rows) - n_real})")
    if skipped:
        log(f"[WARN] без вердикта осталось {skipped} слов — запросы не выполнились")
    s = llm.structured_stats
    log(f"[INFO] структурных ответов: {s['tool']}, разобрано из текста: {s['fallback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
