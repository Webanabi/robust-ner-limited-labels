from __future__ import annotations

import argparse
import json
from src.evaluation.evaluate import evaluate_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_files(args.gold, args.predictions, args.output, args.metadata)
    print(json.dumps(result["metrics"]["micro"], indent=2))


if __name__ == "__main__":
    main()

