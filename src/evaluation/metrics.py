"""Strict exact-boundary, exact-type entity metrics."""
from __future__ import annotations

from collections import Counter
from src.data.dataset import ENTITY_TYPES, spans


def score(gold: list[dict], predicted: list[dict]) -> dict:
    if len(gold) != len(predicted):
        raise ValueError("Gold/prediction lengths differ")
    correct: Counter = Counter()
    gold_count: Counter = Counter()
    pred_count: Counter = Counter()
    for truth, guess in zip(gold, predicted):
        if truth["id"] != guess["id"] or truth["tokens"] != guess["tokens"]:
            raise ValueError(f"Prediction id/tokens do not match gold: {truth['id']}")
        actual = set(spans(truth["ner_tags"]))
        selected = set(spans(guess["pred_tags"], repair=True))
        gold_count.update(kind for _, _, kind in actual)
        pred_count.update(kind for _, _, kind in selected)
        correct.update(kind for _, _, kind in actual & selected)

    def rates(tp: int, actual: int, guessed: int) -> dict:
        precision = tp / guessed if guessed else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1,
                "correct": tp, "gold": actual, "predicted": guessed}

    return {"micro": rates(sum(correct.values()), sum(gold_count.values()), sum(pred_count.values())),
            "per_type": {kind: rates(correct[kind], gold_count[kind], pred_count[kind])
                         for kind in ENTITY_TYPES},
            "sentences": len(gold)}

