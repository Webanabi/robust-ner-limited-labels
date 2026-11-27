"""Shared experiment bookkeeping; contains no model-specific training logic."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any


def safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("run_id may contain only letters, digits, underscore, dot and hyphen")
    return value


def fraction_name(fraction: float) -> str:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    percent = fraction * 100
    if abs(percent - round(percent)) > 1e-9:
        raise ValueError("fraction must be a whole percentage for shared file names")
    return f"{round(percent)}pct"


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Config must be JSON or YAML")
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardware_snapshot() -> dict[str, Any]:
    result = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch
    except ImportError:
        return result
    result["torch"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def check_manifest(rows: list[dict], manifest_path: str | Path,
                   fraction: float, seed: int) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["seed"] != seed or abs(manifest["fraction"] - fraction) > 1e-9:
        raise ValueError(f"{manifest_path}: seed/fraction mismatch")
    if manifest["sentence_ids"] != [row["id"] for row in rows]:
        raise ValueError(f"{manifest_path}: sentence IDs differ from training JSONL")
    if manifest["word_count"] != sum(len(row["tokens"]) for row in rows):
        raise ValueError(f"{manifest_path}: word count differs from training JSONL")


def check_noisy_pair(clean: list[dict], noisy: list[dict]) -> None:
    if len(clean) != len(noisy):
        raise ValueError("Clean/noisy training sentence counts differ")
    for original, changed in zip(clean, noisy):
        if (original["id"] != changed["id"]
                or original["ner_tags"] != changed["ner_tags"]
                or len(original["tokens"]) != len(changed["tokens"])):
            raise ValueError(f"Clean/noisy alignment mismatch at {original['id']}")


def update_matrix(path: str | Path | None, model_id: str, fraction: float, seed: int,
                  status: str, checkpoint_path: str | Path,
                  condition: str | None = None, prediction_path: str | Path | None = None,
                  metadata_path: str | Path | None = None,
                  hyperparameters: dict | None = None) -> int:
    if path is None or not Path(path).exists():
        return 0
    matrix = Path(path)
    with matrix.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"Empty experiment matrix: {matrix}")
    changed = 0
    for row in rows:
        if (row["model_id"] == model_id and abs(float(row["fraction"]) - fraction) < 1e-9
                and int(row["seed"]) == seed and (condition is None or row["condition"] == condition)):
            row["status"] = status
            row["checkpoint_path"] = str(checkpoint_path)
            if prediction_path is not None:
                row["prediction_path"] = str(prediction_path)
            if metadata_path is not None:
                row["metadata_path"] = str(metadata_path)
            if hyperparameters is not None:
                row["hyperparameters"] = json.dumps(hyperparameters, ensure_ascii=False, sort_keys=True)
            changed += 1
    if changed:
        temporary = matrix.with_suffix(matrix.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(matrix)
    return changed

