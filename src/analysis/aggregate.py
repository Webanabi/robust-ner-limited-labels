"""Aggregate only completed, measured evaluation files."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def aggregate(paths: list[str | Path], output_dir: str | Path) -> list[dict]:
    records = []
    for path in paths:
        item = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = item.get("metadata", {})
        if not {"model_id", "fraction", "seed", "condition"} <= meta.keys():
            raise ValueError(f"{path}: missing experiment metadata")
        metric = item["metrics"]
        record = {key: meta[key] for key in ("model_id", "fraction", "seed", "condition")}
        record.update({"f1": metric["micro"]["f1"], "precision": metric["micro"]["precision"],
                       "recall": metric["micro"]["recall"],
                       "training_seconds": meta.get("training_seconds"),
                       "inference_seconds": meta.get("inference_seconds"),
                       "sentence_count": metric["sentences"],
                       "latency_ms_per_sentence": (
                           1000 * meta["inference_seconds"] / metric["sentences"]
                           if meta.get("inference_seconds") is not None and metric["sentences"] else None)})
        record.update({f"f1_{kind}": values["f1"] for kind, values in metric["per_type"].items()})
        record.update({f"error_{kind}": item.get("error_counts", {}).get(kind, 0)
                       for kind in ("missed", "false", "wrong_type", "boundary")})
        records.append(record)
    if not records:
        raise ValueError("No evaluation files provided")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with (target / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(records)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in records:
        groups[(row["model_id"], row["fraction"], row["condition"])].append(row["f1"])
    summaries = [{"model_id": k[0], "fraction": k[1], "condition": k[2],
                  "n_seeds": len(values), "f1_mean": mean(values),
                  "f1_sd": stdev(values) if len(values) >= 2 else None}
                 for k, values in sorted(groups.items())]
    with (target / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, summaries[0])
        writer.writeheader()
        writer.writerows(summaries)
    by_run = {(r["model_id"], r["fraction"], r["seed"], r["condition"]): r for r in records}
    drops = []
    for row in records:
        if row["condition"] == "clean":
            continue
        clean = by_run.get((row["model_id"], row["fraction"], row["seed"], "clean"))
        if clean:
            drops.append({"model_id": row["model_id"], "fraction": row["fraction"],
                          "seed": row["seed"], "condition": row["condition"],
                          "f1_drop": clean["f1"] - row["f1"]})
    with (target / "drops.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["model_id", "fraction", "seed", "condition", "f1_drop"])
        writer.writeheader()
        writer.writerows(drops)
    costs = []
    for row in records:
        if row["condition"] == "clean":
            costs.append({key: row[key] for key in
                          ("model_id", "fraction", "seed", "training_seconds",
                           "inference_seconds", "latency_ms_per_sentence", "sentence_count", "f1")})
    with (target / "costs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["model_id", "fraction", "seed", "training_seconds",
                                         "inference_seconds", "latency_ms_per_sentence",
                                         "sentence_count", "f1"])
        writer.writeheader()
        writer.writerows(costs)
    return records

