from __future__ import annotations

import random
from .dataset import spans


def sample_sentences(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1:
        return list(rows)
    count = max(1, round(len(rows) * fraction))
    selected = set(random.Random(seed).sample(range(len(rows)), count))
    return [row for i, row in enumerate(rows) if i in selected]


def sample_manifest(rows: list[dict], fraction: float, seed: int) -> dict:
    from collections import Counter
    counts = Counter(span[2] for row in rows for span in spans(row["ner_tags"]))
    return {"seed": seed, "fraction": fraction, "sentence_ids": [r["id"] for r in rows],
            "sentence_count": len(rows), "word_count": sum(len(r["tokens"]) for r in rows),
            "entity_count": sum(counts.values()), "entity_types": dict(sorted(counts.items()))}

