"""Shared export for trained models; no substitute predictions."""
from __future__ import annotations

import json
import time
from pathlib import Path
from src.data.dataset import repair_iob2, write_jsonl
from .base import NERModel


def export_predictions(model: NERModel, rows: list[dict], output: str | Path,
                       metadata: dict) -> dict:
    required = {"model_id", "fraction", "seed", "condition", "training_seconds",
                "hardware", "hyperparameters"}
    absent = required - metadata.keys()
    if absent:
        raise ValueError(f"Metadata missing: {sorted(absent)}")
    start = time.perf_counter()
    predictions = [{"id": row["id"], "tokens": row["tokens"],
                    "pred_tags": repair_iob2(model.predict_checked(row["tokens"]))}
                   for row in rows]
    elapsed = time.perf_counter() - start
    write_jsonl(output, predictions, "prediction")
    full = {**metadata, "inference_seconds": elapsed, "sentences": len(rows),
            "words": sum(len(row["tokens"]) for row in rows)}
    target = Path(output)
    target.with_name(target.stem + "_metadata.json").write_text(
        json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    return full

