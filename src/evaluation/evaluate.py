from __future__ import annotations

import csv
import json
from pathlib import Path
from src.data.dataset import read_jsonl
from .metrics import score
from .error_analysis import analyze


def evaluate_files(gold_path: str | Path, predictions_path: str | Path,
                   output_dir: str | Path, metadata_path: str | Path | None = None) -> dict:
    gold = read_jsonl(gold_path, "sentence")
    predictions = read_jsonl(predictions_path, "prediction")
    by_id = {row["id"]: row for row in predictions}
    expected = {row["id"] for row in gold}
    missing, extra = expected - by_id.keys(), by_id.keys() - expected
    if missing or extra:
        raise ValueError(f"Prediction coverage mismatch: {len(missing)} missing, {len(extra)} extra; "
                         f"sample missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")
    ordered = [by_id[row["id"]] for row in gold]
    metrics = score(gold, ordered)
    errors = analyze(gold, ordered)
    metadata = {}
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        required = {"model_id", "fraction", "seed", "condition", "training_seconds",
                    "inference_seconds", "hardware", "hyperparameters"}
        absent = required - metadata.keys()
        if absent:
            raise ValueError(f"Metadata missing: {sorted(absent)}")
    result = {"metrics": metrics, "error_counts": errors["counts"], "metadata": metadata,
              "gold_path": str(gold_path), "prediction_path": str(predictions_path)}
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (target / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    with (target / "evaluation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("type", "precision", "recall", "f1",
                                                     "correct", "gold", "predicted"))
        writer.writeheader()
        writer.writerow({"type": "micro", **metrics["micro"]})
        for kind, values in metrics["per_type"].items():
            writer.writerow({"type": kind, **values})
    with (target / "errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "tokens", "category", "gold", "predicted"))
        writer.writeheader()
        writer.writerows(errors["cases"])
    return result

