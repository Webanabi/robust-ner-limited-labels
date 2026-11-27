"""Run fixed validation/test conditions with a saved model and strict evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from src.data.dataset import read_jsonl
from src.evaluation.evaluate import evaluate_files
from src.experiment import hardware_snapshot, safe_run_id, update_matrix
from src.models.predict import export_predictions
from src.models.registry import load_model

CONDITIONS = ("clean", "lowercase", "typo10", "typo20")


def run_testing(run_id: str, split: str, data_dir: str | Path,
                checkpoint_root: str | Path, predictions_root: str | Path,
                metrics_root: str | Path, conditions: tuple[str, ...] = CONDITIONS,
                matrix_path: str | Path | None = None, model_loader=load_model) -> dict:
    run_id = safe_run_id(run_id)
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    if not conditions or any(c not in CONDITIONS for c in conditions):
        raise ValueError(f"conditions must be among {CONDITIONS}")
    checkpoint = Path(checkpoint_root) / run_id
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No trained model manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "trained" or manifest.get("run_id") != run_id:
        raise ValueError(f"{manifest_path}: not a trained run")
    inputs = {condition: Path(data_dir) / f"{split}_{condition}.jsonl"
              for condition in conditions}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    model_id = manifest["model_id"]
    model = model_loader(model_id, checkpoint)
    model.model_id = model_id
    summary = {"run_id": run_id, "split": split, "results": {}}
    hardware = hardware_snapshot()
    for condition, gold_path in inputs.items():
        rows = read_jsonl(gold_path)
        output = Path(predictions_root) / run_id / f"{split}_{condition}.jsonl"
        metadata = {
            "model_id": model_id, "run_id": run_id, "fraction": manifest["fraction"],
            "seed": manifest["seed"], "condition": condition,
            "training_seconds": manifest["training_seconds"],
            "training_hardware": manifest["hardware"], "hardware": hardware,
            "inference_device": str(getattr(model, "device", "unspecified")),
            "hyperparameters": manifest["hyperparameters"],
            "checkpoint": str(checkpoint), "split": split,
        }
        export_predictions(model, rows, output, metadata)
        metadata_path = output.with_name(output.stem + "_metadata.json")
        metric_dir = Path(metrics_root) / run_id / f"{split}_{condition}"
        result = evaluate_files(gold_path, output, metric_dir, metadata_path)
        summary["results"][condition] = result["metrics"]["micro"]
        if split == "test":
            update_matrix(matrix_path, model_id, manifest["fraction"], manifest["seed"],
                          "evaluated", checkpoint, condition, output, metadata_path,
                          manifest["hyperparameters"])
    summary_path = Path(metrics_root) / run_id / f"{split}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--predictions-root", type=Path, default=Path("results/predictions"))
    parser.add_argument("--metrics-root", type=Path, default=Path("results/metrics"))
    parser.add_argument("--matrix", type=Path, default=Path("EXPERIMENT_MATRIX.csv"))
    args = parser.parse_args()
    summary = run_testing(args.run_id, args.split, args.data_dir, args.checkpoint_root,
                          args.predictions_root, args.metrics_root, tuple(args.conditions), args.matrix)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

