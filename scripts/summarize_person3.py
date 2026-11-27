"""Print the measured Person 3 runs without including corpus text or predictions."""
import json
from pathlib import Path

from src.models.distilbert.model import METHODS
from scripts.run_person3 import CONDITIONS


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main():
    rows = []
    training_count = 0
    evaluation_count = 0
    for fraction in (10, 100):
        for model in METHODS:
            run_id = f"{model}_{fraction}pct_seed42"
            manifest = read(Path("checkpoints") / run_id / "manifest.json")
            trained = bool(manifest and manifest.get("status") == "trained")
            training_count += trained
            row = dict(run_id=run_id, trained=trained, test={})
            if trained:
                report = manifest["training_report"]
                row.update(training_seconds=manifest["training_seconds"], best=report["best"],
                           gpu_name=report["gpu_name"],
                           peak_cuda_allocated_bytes=report["peak_cuda_allocated_bytes"])
                for condition in CONDITIONS:
                    path = Path("results/metrics") / run_id / f"test_{condition}" / "evaluation.json"
                    record = read(path)
                    meta_path = Path("results/predictions") / run_id / f"test_{condition}_metadata.json"
                    metadata = read(meta_path)
                    if record and metadata:
                        row["test"][condition] = dict(f1=record["metrics"]["micro"]["f1"],
                                                      inference_seconds=metadata["inference_seconds"])
                        evaluation_count += 1
            rows.append(row)
    print(json.dumps(dict(trained=training_count, expected_trainings=8,
                          evaluated=evaluation_count, expected_evaluations=32,
                          runs=rows), indent=2))


if __name__ == "__main__":
    main()
