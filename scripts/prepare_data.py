from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import yaml
from src.data.dataset import write_jsonl
from src.data.preprocess import read_conll, read_huggingface
from src.data.sampling import sample_manifest, sample_sentences
from src.data.noise import perturb, training_pair, audit_summary
from src.data.statistics import describe

EXPECTED = {"train": 14041, "validation": 3250, "test": 3453}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else config["seeds"][0]
    fraction = args.fraction if args.fraction is not None else min(config["training_fractions"])
    noise = config["noise"]
    if args.source == "local" and not all((args.train, args.validation, args.test)):
        parser.error("--source local requires --train --validation --test")
    source_files = {"train": args.train, "validation": args.validation, "test": args.test}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "validation", "test"):
        rows = (read_conll(source_files[split], split) if args.source == "local"
                else read_huggingface(split))
        write_jsonl(output / f"{split}.jsonl", rows)
        summary[split] = describe(rows)
        summary[split]["proposal_expected_sentences"] = EXPECTED[split]
        summary[split]["count_matches_proposal"] = len(rows) == EXPECTED[split]
        (output / f"{split}_statistics.json").write_text(
            json.dumps(summary[split], indent=2), encoding="utf-8")
        if split == "train":
            (output / f"train_100pct_seed{seed}_manifest.json").write_text(
                json.dumps(sample_manifest(rows, 1.0, seed), indent=2), encoding="utf-8")
            full_noisy = []
            full_audit = []
            for row in rows:
                new, detail = training_pair(
                    row, seed, noise["lowercase_rate"], noise["typo_rate"],
                    noise["lower_probability"])
                full_noisy.append(new)
                full_audit.append(detail)
            full_name = f"train_100pct_seed{seed}_noisy"
            write_jsonl(output / f"{full_name}.jsonl", full_noisy)
            (output / f"{full_name}_audit.json").write_text(
                json.dumps(audit_summary(full_audit, [
                    {"id": old["id"], "original": old["tokens"], "perturbed": new["tokens"]}
                    for old, new in zip(rows, full_noisy) if old["tokens"] != new["tokens"]
                ]), indent=2), encoding="utf-8")
            selected = sample_sentences(rows, fraction, seed)
            subset_name = f"train_{int(fraction * 100)}pct_seed{seed}"
            write_jsonl(output / f"{subset_name}.jsonl", selected)
            manifest = sample_manifest(selected, fraction, seed)
            (output / f"{subset_name}_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")
            selected_ids = {row["id"] for row in selected}
            noisy = [new for new in full_noisy if new["id"] in selected_ids]
            audit = [item for item in full_audit if item["id"] in selected_ids]
            write_jsonl(output / f"{subset_name}_noisy.jsonl", noisy)
            (output / f"{subset_name}_noise_audit.json").write_text(
                json.dumps(audit_summary(audit, [
                    {"id": old["id"], "original": old["tokens"], "perturbed": new["tokens"]}
                    for old, new in zip(selected, noisy) if old["tokens"] != new["tokens"]
                ]), indent=2), encoding="utf-8")
        else:
            for condition in ("clean", "lowercase", "typo10", "typo20"):
                noise_seed = noise["validation_seed"] if split == "validation" else noise["test_seed"]
                variants, audits = zip(*(perturb(row, condition, noise_seed) for row in rows))
                write_jsonl(output / f"{split}_{condition}.jsonl", variants)
                (output / f"{split}_{condition}_noise_audit.json").write_text(
                    json.dumps(audit_summary(list(audits), [
                        {"id": old["id"], "original": old["tokens"], "perturbed": new["tokens"]}
                        for old, new in zip(rows, variants) if old["tokens"] != new["tokens"]
                    ]), indent=2), encoding="utf-8")
    source_hashes = ({split: hashlib.sha256(path.read_bytes()).hexdigest()
                      for split, path in source_files.items()} if args.source == "local" else {})
    (output / "source_audit.json").write_text(
        json.dumps({"source": args.source, "source_sha256": source_hashes,
                    "config": config, "seed": seed, "fraction": fraction,
                    "splits": summary}, indent=2), encoding="utf-8")
    print(json.dumps({k: {"sentences": v["sentences"],
                          "matches_proposal": v["count_matches_proposal"],
                          "document_boundaries_available": v["document_boundaries_available"]}
                      for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()

