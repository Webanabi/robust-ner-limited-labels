from __future__ import annotations

import argparse
import json
from pathlib import Path
from src.data.dataset import read_jsonl
from src.models.registry import load_model
from src.models.predict import export_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-metadata", required=True,
                        help="JSON with model_id,fraction,seed,condition,training_seconds,hardware,hyperparameters")
    args = parser.parse_args()
    metadata = json.loads(Path(args.run_metadata).read_text(encoding="utf-8"))
    if metadata.get("model_id") != args.model_id:
        parser.error("model_id in metadata differs from --model-id")
    rows = read_jsonl(args.input)
    model = load_model(args.model_id, args.checkpoint)
    result = export_predictions(model, rows, args.output, metadata)
    print(json.dumps({"sentences": result["sentences"],
                      "inference_seconds": result["inference_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
