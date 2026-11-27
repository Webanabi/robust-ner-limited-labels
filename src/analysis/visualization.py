"""Plots from measured runs and data statistics; absent inputs stay absent."""
from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path


def generate(tables_dir: str | Path, figures_dir: str | Path,
             statistics_files: list[str | Path] | None = None,
             pca_points: str | Path | None = None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables = Path(tables_dir)
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    with (tables / "runs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No measured runs to plot")
    output = []
    def save(name: str) -> None:
        plt.tight_layout()
        path = figures / name
        plt.savefig(path, dpi=180)
        plt.close()
        output.append(str(path))

    labels = [f"{r['model_id']}\n{r['fraction']} {r['condition']} s{r['seed']}" for r in rows]
    plt.figure(figsize=(max(8, len(rows) * .7), 5))
    plt.bar(range(len(rows)), [float(r["f1"]) for r in rows])
    plt.xticks(range(len(rows)), labels, rotation=70, ha="right", fontsize=8)
    plt.ylabel("Exact entity micro F1")
    plt.ylim(0, 1)
    save("model_f1.png")
    conditions = ("clean", "lowercase", "typo10", "typo20")
    runs = sorted({(r["model_id"], r["fraction"], r["seed"]) for r in rows})
    plt.figure(figsize=(8, 5))
    for model, fraction, seed in runs:
        values = {r["condition"]: float(r["f1"]) for r in rows
                  if (r["model_id"], r["fraction"], r["seed"]) == (model, fraction, seed)}
        if values:
            plt.plot([c for c in conditions if c in values],
                     [values[c] for c in conditions if c in values], marker="o",
                     label=f"{model} {fraction} s{seed}")
    plt.xlabel("Test condition")
    plt.ylabel("Exact entity micro F1")
    plt.ylim(0, 1)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    save("robustness_curves.png")
    clean_rows = [r for r in rows if r["condition"] == "clean"]
    if clean_rows:
        plt.figure(figsize=(8, 5))
        for model in sorted({r["model_id"] for r in clean_rows}):
            subset = [r for r in clean_rows if r["model_id"] == model]
            plt.scatter([float(r["fraction"]) for r in subset],
                        [float(r["f1"]) for r in subset], label=model)
        plt.xlabel("Training fraction")
        plt.ylabel("Clean entity F1")
        plt.xlim(0, 1.05)
        plt.ylim(0, 1)
        plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        save("training_budget.png")
    for field, title in (("fraction", "Training fraction"), ("condition", "Noise condition")):
        keys = sorted({r[field] for r in rows})
        plt.figure(figsize=(max(6, len(keys) * 1.5), 4))
        label_argument = ("tick_labels" if "tick_labels" in inspect.signature(plt.boxplot).parameters
                          else "labels")
        plt.boxplot([[float(r["f1"]) for r in rows if r[field] == key] for key in keys],
                    **{label_argument: keys})
        plt.ylabel("Exact entity micro F1")
        plt.xlabel(title)
        plt.ylim(0, 1)
        save(f"by_{field}.png")
    kinds = ("PER", "ORG", "LOC", "MISC")
    plt.figure(figsize=(max(8, len(rows) * .7), 5))
    for i, kind in enumerate(kinds):
        plt.bar([x + (i - 1.5) * .2 for x in range(len(rows))],
                [float(r[f"f1_{kind}"]) for r in rows], width=.2, label=kind)
    plt.xticks(range(len(rows)), labels, rotation=70, ha="right", fontsize=8)
    plt.legend()
    plt.ylabel("Per-type F1")
    save("entity_types.png")
    clean = {(r["model_id"], r["fraction"], r["seed"]): r for r in rows
             if r["condition"] == "clean"}
    paired = [(r, clean[(r["model_id"], r["fraction"], r["seed"])])
              for r in rows if r["condition"] != "clean"
              and (r["model_id"], r["fraction"], r["seed"]) in clean]
    if paired:
        plt.figure(figsize=(max(8, len(paired) * .7), 5))
        plt.bar(range(len(paired)), [float(c["f1"]) - float(r["f1"]) for r, c in paired])
        plt.xticks(range(len(paired)),
                   [f"{r['model_id']}\n{r['fraction']} {r['condition']}" for r, _ in paired],
                   rotation=70, ha="right", fontsize=8)
        plt.ylabel("Clean - noisy F1")
        save("f1_drop.png")
    neural = [r for r in rows if r["model_id"] in
              ("distilbert_clean", "distilbert_aug",
               "distilbert_consistency", "distilbert_balanced")]
    if neural:
        plt.figure(figsize=(max(7, len(neural) * .6), 5))
        plt.bar(range(len(neural)), [float(r["f1"]) for r in neural])
        plt.xticks(range(len(neural)),
                   [f"{r['model_id']}\n{r['fraction']} {r['condition']}" for r in neural],
                   rotation=70, ha="right", fontsize=8)
        plt.ylabel("Exact entity micro F1")
        plt.ylim(0, 1)
        save("neural_ablation.png")
    costs = [r for r in rows if r["condition"] == "clean" and r["training_seconds"] not in ("", None)]
    if costs:
        plt.figure(figsize=(7, 5))
        for r in costs:
            plt.scatter(float(r["training_seconds"]) / 60, float(r["f1"]))
            plt.annotate(r["model_id"], (float(r["training_seconds"]) / 60, float(r["f1"])), fontsize=7)
        plt.xlabel("Training minutes")
        plt.ylabel("Clean entity F1")
        save("training_cost.png")
    error_kinds = ("missed", "false", "wrong_type", "boundary")
    plt.figure(figsize=(max(8, len(rows) * .7), 5))
    bottoms = [0] * len(rows)
    for kind in error_kinds:
        values = [int(r[f"error_{kind}"]) for r in rows]
        plt.bar(range(len(rows)), values, bottom=bottoms, label=kind)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    plt.xticks(range(len(rows)), labels, rotation=70, ha="right", fontsize=8)
    plt.ylabel("Error count")
    plt.legend()
    save("error_types.png")
    pca_rows = [r for r in rows if r["model_id"] in ("lr_raw", "lr_pca64", "lr_pca128")]
    if pca_rows:
        plt.figure(figsize=(max(7, len(pca_rows) * .7), 5))
        plt.bar(range(len(pca_rows)), [float(r["f1"]) for r in pca_rows])
        plt.xticks(range(len(pca_rows)),
                   [f"{r['model_id']}\n{r['fraction']} {r['condition']}" for r in pca_rows],
                   rotation=70, ha="right", fontsize=8)
        plt.ylabel("Exact entity micro F1")
        plt.ylim(0, 1)
        save("pca_comparison.png")
    if pca_points:
        with Path(pca_points).open(encoding="utf-8", newline="") as stream:
            points = list(csv.DictReader(stream))
        if not points or not {"x", "y", "label"} <= points[0].keys():
            raise ValueError("PCA points CSV needs x,y,label columns")
        plt.figure(figsize=(7, 5))
        for kind in sorted({p["label"] for p in points}):
            subset = [p for p in points if p["label"] == kind]
            plt.scatter([float(p["x"]) for p in subset], [float(p["y"]) for p in subset],
                        s=6, alpha=.5, label=kind)
        plt.xlabel("PCA component 1")
        plt.ylabel("PCA component 2")
        plt.legend()
        save("pca_2d.png")
    if statistics_files:
        stats = {Path(p).stem: json.loads(Path(p).read_text(encoding="utf-8"))
                 for p in statistics_files}
        for field, title in (("entity_types", "Entity counts"),
                             ("sentence_lengths", "Sentence length"),
                             ("entity_lengths", "Entity length")):
            plt.figure(figsize=(8, 4))
            for name, item in stats.items():
                values = item[field]
                plt.plot(list(values), list(values.values()), marker="o", label=name)
            plt.xlabel(title)
            plt.ylabel("Count")
            plt.legend()
            save(f"data_{field}.png")
    return output
