"""One-to-one error matching: exact, wrong type, boundary, missed/false."""
from __future__ import annotations

from collections import Counter
from src.data.dataset import spans


def _overlap(a: tuple, b: tuple) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def analyze(gold: list[dict], predictions: list[dict]) -> dict:
    cases: list[dict] = []
    counts: Counter = Counter()
    for row, pred in zip(gold, predictions):
        actual = sorted(spans(row["ner_tags"]))
        guessed = sorted(spans(pred["pred_tags"], repair=True))
        remaining_gold = set(range(len(actual)))
        remaining_pred = set(range(len(guessed)))

        def record(kind: str, gi: int | None, pi: int | None) -> None:
            counts[kind] += 1
            cases.append({"id": row["id"], "tokens": row["tokens"], "category": kind,
                          "gold": actual[gi] if gi is not None else None,
                          "predicted": guessed[pi] if pi is not None else None})
            if gi is not None:
                remaining_gold.remove(gi)
            if pi is not None:
                remaining_pred.remove(pi)

        for gi in sorted(list(remaining_gold)):
            match = next((pi for pi in sorted(remaining_pred) if guessed[pi] == actual[gi]), None)
            if match is not None:
                remaining_gold.remove(gi)
                remaining_pred.remove(match)
        for gi in sorted(list(remaining_gold)):
            match = next((pi for pi in sorted(remaining_pred)
                          if guessed[pi][:2] == actual[gi][:2]), None)
            if match is not None:
                record("wrong_type", gi, match)
        candidates = sorted(
            ((-_overlap(actual[gi], guessed[pi]), gi, pi)
             for gi in remaining_gold for pi in remaining_pred
             if _overlap(actual[gi], guessed[pi]) > 0)
        )
        for _, gi, pi in candidates:
            if gi in remaining_gold and pi in remaining_pred:
                record("boundary", gi, pi)
        for gi in sorted(list(remaining_gold)):
            record("missed", gi, None)
        for pi in sorted(list(remaining_pred)):
            record("false", None, pi)
    return {"counts": dict(counts), "cases": cases}

