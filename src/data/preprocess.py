"""Read local CoNLL files, preserving -DOCSTART- boundaries."""
from __future__ import annotations

from pathlib import Path
from .dataset import repair_iob2, validate_sentence


def read_conll(path: str | Path, split: str) -> list[dict]:
    rows: list[dict] = []
    tokens: list[str] = []
    tags: list[str] = []
    document = 0
    saw_docstart = False

    def flush() -> None:
        nonlocal tokens, tags
        if tokens:
            converted = repair_iob2(tags)
            row = {"id": f"{split}_{len(rows):06d}", "doc_id": f"{split}_doc_{document:06d}",
                   "tokens": tokens, "ner_tags": converted,
                   "source_tag_conversions": sum(a != b for a, b in zip(tags, converted))}
            validate_sentence(row)
            rows.append(row)
            tokens, tags = [], []

    with Path(path).open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                flush()
                continue
            columns = line.split()
            if columns[0] == "-DOCSTART-":
                flush()
                document += 1
                saw_docstart = True
                continue
            if len(columns) < 2:
                raise ValueError(f"{path}:{number}: expected token and NER tag")
            tokens.append(columns[0])
            tags.append(columns[-1])
    flush()
    if not rows:
        raise ValueError(f"No sentences in {path}")
    if not saw_docstart:
        raise ValueError(f"{path}: no -DOCSTART- markers; document grouping unavailable")
    return rows


def read_huggingface(split: str) -> list[dict]:
    """Optional download. Source lacks trusted document boundaries; flag this."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets from requirements-gpu.txt") from exc
    dataset = load_dataset("eriktks/conll2003", split=split, trust_remote_code=True)
    names = dataset.features["ner_tags"].feature.names
    rows = []
    for i, item in enumerate(dataset):
        row = {"id": f"{split}_{i:06d}", "doc_id": "UNAVAILABLE",
               "tokens": list(item["tokens"]), "ner_tags": [names[x] for x in item["ner_tags"]]}
        validate_sentence(row)
        rows.append(row)
    return rows

