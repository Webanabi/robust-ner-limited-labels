"""Create a reviewable, unexecuted experiment register from config."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import yaml

METHODS = {
    "crf": "crf", "crf_no_case": "crf_ablation_case",
    "crf_no_affix": "crf_ablation_affix",
    "lr_raw": "frozen_lr", "lr_pca64": "frozen_pca64_lr",
    "lr_pca128": "frozen_pca128_lr",
    "distilbert_clean": "clean_only",
    "distilbert_aug": "noise_augmentation",
    "distilbert_consistency": "ordinary_consistency",
    "distilbert_balanced": "entity_balanced_consistency",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--output", type=Path, default=Path("EXPERIMENT_MATRIX.csv"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; edit statuses directly or choose another output")
    plan = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    for tier in ("core", "supplemental", "repeat", "optional"):
        spec = plan[tier]
        for model in spec["models"]:
            for fraction in spec["fractions"]:
                for seed in spec["seeds"]:
                    for condition in plan["conditions"]:
                        rows.append({"tier": tier, "model_id": model, "fraction": fraction,
                                     "seed": seed, "condition": condition, "method": METHODS[model],
                                     "hyperparameters": json.dumps({}), "status": "planned",
                                     "checkpoint_path": "", "prediction_path": "",
                                     "metadata_path": "", "notes": ""})
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} planned rows to {args.output}")


if __name__ == "__main__":
    main()

