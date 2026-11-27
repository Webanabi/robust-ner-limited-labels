from __future__ import annotations

import argparse
from pathlib import Path
from src.analysis.aggregate import aggregate
from src.analysis.visualization import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", nargs="+", type=Path, required=True)
    parser.add_argument("--statistics", nargs="*", type=Path)
    parser.add_argument("--pca-points", type=Path)
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    aggregate(args.evaluations, args.tables)
    for path in generate(args.tables, args.figures, args.statistics, args.pca_points):
        print(path)


if __name__ == "__main__":
    main()

