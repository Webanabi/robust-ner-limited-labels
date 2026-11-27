"""Run the seed-42 DistilBERT experiment matrix through the shared CLIs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.models.distilbert.model import METHODS

CONDITIONS = ("clean", "lowercase", "typo10", "typo20")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", nargs="+", type=float, choices=(0.1, 1.0), default=[0.1, 1.0])
    parser.add_argument("--models", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--hyperparameters", type=Path)
    args = parser.parse_args()
    for fraction in sorted(set(args.fractions)):
        for model in args.models:
            run_id = f"{model}_{int(fraction * 100)}pct_seed42"
            checkpoint = Path("checkpoints") / run_id
            manifest_path = checkpoint / "manifest.json"
            if checkpoint.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected = dict(run_id=run_id, status="trained", model_id=model, fraction=fraction, seed=42)
                if any(manifest.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Incompatible existing checkpoint at {checkpoint}")
            else:
                command = ["-m", "scripts.train_model", "--model-id", model,
                           "--fraction", str(fraction), "--seed", "42"]
                if args.hyperparameters:
                    command.extend(["--hyperparameters", str(args.hyperparameters)])
                subprocess.run([sys.executable, *command], check=True)
            root = Path("results/metrics") / run_id
            summary_path = root / "test_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                complete = (set(summary.get("results", {})) == set(CONDITIONS)
                            and all((root / f"test_{condition}" / "evaluation.json").is_file()
                                    for condition in CONDITIONS))
                if complete:
                    print(f"{run_id}: four existing test evaluations", flush=True)
                    continue
            print(f"{run_id}: evaluating four test conditions", flush=True)
            subprocess.run([sys.executable, "-m", "scripts.test_model", "--run-id", run_id], check=True)


if __name__ == "__main__":
    main()
