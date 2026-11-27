"""Shared sentence and prediction formats. No training dependency."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

LABELS = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")


def validate_tags(tags: list[str], *, strict_bio: bool = True) -> None:
    previous = "O"
    for i, tag in enumerate(tags):
        if tag not in LABEL_TO_ID:
            raise ValueError(f"Illegal tag at word {i}: {tag!r}")
        if strict_bio and tag.startswith("I-") and previous not in (tag, "B-" + tag[2:]):
            raise ValueError(f"Invalid IOB2 continuation at word {i}: {tag}")
        previous = tag


def repair_iob2(tags: list[str]) -> list[str]:
    """Shared prediction repair: invalid I-X becomes B-X."""
    validate_tags(tags, strict_bio=False)
    result: list[str] = []
    for tag in tags:
        if tag.startswith("I-") and (not result or result[-1] not in (tag, "B-" + tag[2:])):
            result.append("B-" + tag[2:])
        else:
            result.append(tag)
    return result


def validate_sentence(row: dict) -> None:
    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("Sentence needs a nonempty string id")
    if not isinstance(row.get("doc_id"), str) or not row["doc_id"]:
        raise ValueError(f"{row['id']}: sentence needs doc_id")
    tokens, tags = row.get("tokens"), row.get("ner_tags")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) and t for t in tokens):
        raise ValueError(f"{row['id']}: tokens must be nonempty strings")
    if not isinstance(tags, list) or len(tags) != len(tokens):
        raise ValueError(f"{row['id']}: token/tag lengths differ")
    validate_tags(tags)


def validate_prediction(row: dict) -> None:
    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("Prediction needs a nonempty string id")
    tokens, tags = row.get("tokens"), row.get("pred_tags")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) and t for t in tokens):
        raise ValueError(f"{row['id']}: invalid tokens")
    if not isinstance(tags, list) or len(tags) != len(tokens):
        raise ValueError(f"{row['id']}: token/prediction lengths differ")
    validate_tags(tags, strict_bio=False)


def read_jsonl(path: str | Path, kind: str = "sentence") -> list[dict]:
    validator = validate_sentence if kind == "sentence" else validate_prediction
    if kind not in ("sentence", "prediction"):
        raise ValueError("kind must be sentence or prediction")
    rows: list[dict] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validator(row)
                if row["id"] in seen:
                    raise ValueError(f"Duplicate id {row['id']}")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict], kind: str = "sentence") -> None:
    validator = validate_sentence if kind == "sentence" else validate_prediction
    if kind not in ("sentence", "prediction"):
        raise ValueError("kind must be sentence or prediction")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            validator(row)
            if row["id"] in seen:
                raise ValueError(f"Duplicate id {row['id']}")
            seen.add(row["id"])
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def spans(tags: list[str], *, repair: bool = False) -> list[tuple[int, int, str]]:
    """Return half-open word spans (start, end, type)."""
    sequence = repair_iob2(tags) if repair else tags
    validate_tags(sequence)
    found: list[tuple[int, int, str]] = []
    start: int | None = None
    entity_type: str | None = None
    for i, tag in enumerate(sequence + ["O"]):
        kind = None if tag == "O" else tag[2:]
        begins = tag.startswith("B-")
        if start is not None and (kind != entity_type or begins):
            found.append((start, i, entity_type))
            start, entity_type = None, None
        if kind is not None and start is None:
            start, entity_type = i, kind
    return found

