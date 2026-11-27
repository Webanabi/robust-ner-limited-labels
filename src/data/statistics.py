from __future__ import annotations

from collections import Counter
from statistics import mean
from .dataset import spans


def describe(rows: list[dict]) -> dict:
    entities = [span for row in rows for span in spans(row["ner_tags"])]
    types = Counter(span[2] for span in entities)
    lengths = Counter(len(row["tokens"]) for row in rows)
    entity_lengths = Counter(end - start for start, end, _ in entities)
    return {"sentences": len(rows), "words": sum(len(row["tokens"]) for row in rows),
            "documents": len({row["doc_id"] for row in rows}),
            "document_boundaries_available": all(row["doc_id"] != "UNAVAILABLE" for row in rows),
            "source_tag_conversions": sum(row.get("source_tag_conversions", 0) for row in rows),
            "entities": len(entities), "entity_types": dict(sorted(types.items())),
            "sentence_lengths": dict(sorted(lengths.items())),
            "entity_lengths": dict(sorted(entity_lengths.items())),
            "mean_sentence_length": mean(len(row["tokens"]) for row in rows) if rows else 0}
