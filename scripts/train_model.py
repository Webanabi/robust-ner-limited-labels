"""Train one registered model once. Model owners implement train/save."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from src.data.dataset import read_jsonl
from src.experiment import (check_manifest, check_noisy_pair, fraction_name,
                            hardware_snapshot, load_config, safe_run_id,
                            sha256_file, update_matrix)
from src.models.registry import create_model


def run_training(model_id: str, fraction: float, seed: int, data_dir: str | Path,
                 checkpoint_root: str | Path, run_id: str, hyperparameters: dict,
                 defaults: dict, matrix_path: str | Path | None = None,
                 model_factory=create_model) -> dict:
    run_id = safe_run_id(run_id)
    name = fraction_name(fraction)
    data_dir = Path(data_dir)
    if fraction == 1:
        train_path = data_dir / "train.jsonl"
    else:
        train_path = data_dir / f"train_{name}_seed{seed}.jsonl"
    manifest_path = data_dir / f"train_{name}_seed{seed}_manifest.json"
    noisy_path = data_dir / f"train_{name}_seed{seed}_noisy.jsonl"
    if fraction == 1:
        noisy_path = data_dir / f"train_100pct_seed{seed}_noisy.jsonl"
    train_rows = read_jsonl(train_path)
    check_manifest(train_rows, manifest_path, fraction, seed)
    if model_id.startswith(("crf", "lr_")):
        documents = {row["doc_id"] for row in train_rows}
        if "UNAVAILABLE" in documents or len(documents) < 3:
            raise ValueError("CRF/LR model selection needs at least three real document groups")
    noisy_rows = read_jsonl(noisy_path)
    check_noisy_pair(train_rows, noisy_rows)
    validation = read_jsonl(data_dir / "validation.jsonl")
    variants = {condition: read_jsonl(data_dir / f"validation_{condition}.jsonl")
                for condition in ("clean", "lowercase", "typo10")}
    for condition, rows in variants.items():
        if [r["id"] for r in rows] != [r["id"] for r in validation]:
            raise ValueError(f"Validation {condition}: IDs differ from clean validation")
        check_noisy_pair(validation, rows)
    output = Path(checkpoint_root) / run_id
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new run_id")
    selection_rule = (
        "three-fold GroupKFold by doc_id within training subset"
        if model_id.startswith(("crf", "lr_"))
        else "mean entity F1 on validation clean/lowercase/typo10"
    )
    runtime_config = {
        "model_id": model_id, "fraction": fraction, "seed": seed,
        "hyperparameters": hyperparameters, "defaults": defaults,
        "noisy_train_rows": noisy_rows, "validation_variants": variants,
        "selection_rule": selection_rule,
    }
    model = model_factory(model_id)
    model.model_id = model_id
    started = time.perf_counter()
    report = model.train(train_rows, validation, runtime_config)
    training_seconds = time.perf_counter() - started
    if report is not None and not isinstance(report, dict):
        raise ValueError("model.train must return None or a JSON-compatible dict")
    output.mkdir(parents=True)
    model.save(output)
    if not any(path.is_file() for path in output.rglob("*")):
        raise ValueError(f"{model_id}.save created no checkpoint files in {output}")
    record = {
        "status": "trained", "run_id": run_id, "model_id": model_id,
        "fraction": fraction, "seed": seed, "training_seconds": training_seconds,
        "hardware": hardware_snapshot(), "hyperparameters": hyperparameters,
        "defaults": defaults,
        "selection_rule": runtime_config["selection_rule"],
        "checkpoint": str(output), "train_data": str(train_path),
        "train_data_sha256": sha256_file(train_path),
        "noisy_data": str(noisy_path), "noisy_data_sha256": sha256_file(noisy_path),
        "validation_data_sha256": sha256_file(data_dir / "validation.jsonl"),
        "training_report": report or {},
    }
    record_path = output / "manifest.json"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    update_matrix(matrix_path, model_id, fraction, seed, "trained", output,
                  metadata_path=record_path, hyperparameters=hyperparameters)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-id")
    parser.add_argument("--hyperparameters", type=Path, help="Optional JSON or YAML object")
    parser.add_argument("--defaults", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--matrix", type=Path, default=Path("EXPERIMENT_MATRIX.csv"))
    args = parser.parse_args()
    run_id = args.run_id or f"{args.model_id}_{fraction_name(args.fraction)}_seed{args.seed}"
    try:
        result = run_training(args.model_id, args.fraction, args.seed, args.data_dir,
                              args.checkpoint_root, run_id,
                              load_config(args.hyperparameters) if args.hyperparameters else {},
                              load_config(args.defaults), args.matrix)
    except NotImplementedError as exc:
        parser.exit(2, f"{args.model_id}: {exc}\n")
    print(json.dumps({"run_id": result["run_id"], "checkpoint": result["checkpoint"],
                      "training_seconds": result["training_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

