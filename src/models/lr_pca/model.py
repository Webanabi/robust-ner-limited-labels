"""Frozen DistilBERT word features and a document-selected logistic regression."""
from __future__ import annotations

import json
import math
from importlib.metadata import version
from pathlib import Path

from src.data.dataset import ID_TO_LABEL, LABELS, LABEL_TO_ID, repair_iob2, validate_sentence
from src.evaluation.metrics import score
from src.models.base import NERModel
from .encoder import FrozenEncoder, MODEL_NAME, MODEL_REVISION


def document_folds(train_rows: list[dict]):
    """Split sentence indices before expanding any fold into its word indices."""
    from sklearn.model_selection import GroupKFold

    groups = [row.get("doc_id") for row in train_rows]
    if any(not isinstance(g, str) or not g.strip() or g == "UNAVAILABLE" for g in groups):
        raise ValueError("LR grouped CV requires a real document ID for every sentence")
    if len(set(groups)) < 3:
        raise ValueError("LR grouped CV needs at least three real documents")
    return list(GroupKFold(n_splits=3).split(train_rows, groups=groups))


def _decode(label_ids) -> list[str]:
    return repair_iob2([ID_TO_LABEL[i] for i in label_ids])


def _cross_validate(rows, features, labels, folds, c_values, lr_options):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    lengths = [len(row["tokens"]) for row in rows]
    results = [{"C": c, "fold_f1": []} for c in c_values]
    for fit_sentences, heldout_sentences in folds:
        # Boolean indexing preserves the original sentence/word order for scoring.
        fit_mask = np.zeros(len(rows), dtype=bool)
        fit_mask[fit_sentences] = True
        fit_words = np.repeat(fit_mask, lengths)
        fit_x, fit_y = features[fit_words], labels[fit_words]
        heldout_x = features[~fit_words]
        gold = [rows[i] for i in sorted(heldout_sentences)]
        offsets = np.cumsum([len(row["tokens"]) for row in gold])[:-1]
        for result in results:
            estimator = LogisticRegression(C=result["C"], **lr_options)
            estimator.fit(fit_x, fit_y)
            predictions = np.split(estimator.predict(heldout_x), offsets)
            guesses = [{"id": row["id"], "tokens": row["tokens"], "pred_tags": _decode(ids)}
                       for row, ids in zip(gold, predictions)]
            result["fold_f1"].append(score(gold, guesses)["micro"]["f1"])
    for result in results:
        result["mean_f1"] = sum(result["fold_f1"]) / len(folds)
    return results


def _default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class FrozenFeatureModel(NERModel):
    model_id = "lr_raw"

    def __init__(self):
        self.encoder = None
        self.classifier = None
        self.report = {}
        self.batch_size = 8

    @property
    def device(self):
        return self.encoder.device if self.encoder is not None else None

    def _require_raw(self):
        if self.model_id != "lr_raw":
            raise NotImplementedError(f"{self.model_id} is a deferred PCA experiment")

    def train(self, train_rows: list[dict], validation_rows: list[dict], config: dict) -> dict:
        self._require_raw()
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        folds = document_folds(train_rows)
        for row in train_rows:
            validate_sentence(row)
        hyper = config.get("hyperparameters", {})
        c_values = [float(c) for c in hyper.get("c_values", [0.1, 1.0, 10.0])]
        if not c_values or any(not math.isfinite(c) or c <= 0 for c in c_values):
            raise ValueError("c_values must contain finite positive values")
        batch_size = int(hyper.get("batch_size", 8))
        max_iter = int(hyper.get("max_iter", 300))
        if min(batch_size, max_iter) < 1:
            raise ValueError("batch_size and max_iter must be positive")
        lr_options = {"solver": "lbfgs", "max_iter": max_iter, "random_state": config.get("seed", 42)}

        encoder = FrozenEncoder.pretrained(device=hyper.get("device") or _default_device())
        sentences = [row["tokens"] for row in train_rows]
        features = encoder.extract(sentences, batch_size=batch_size)
        labels = np.array([LABEL_TO_ID[tag] for row in train_rows for tag in row["ner_tags"]])
        if len(features) != len(labels):
            raise ValueError("Frozen encoder must return one feature vector per word")
        results = _cross_validate(train_rows, features, labels, folds, c_values, lr_options)
        best = max(results, key=lambda result: result["mean_f1"])
        classifier = LogisticRegression(C=best["C"], **lr_options)
        classifier.fit(features, labels)

        self.encoder = encoder
        self.classifier = classifier
        self.batch_size = batch_size
        self.report = {
            "encoder": MODEL_NAME, "encoder_revision": MODEL_REVISION,
            "feature_device": self.device, "classifier_device": "cpu", "batch_size": batch_size,
            "training_documents": len({row["doc_id"] for row in train_rows}),
            "training_words": len(labels), "feature_dimension": features.shape[1],
            "selection_metric": "mean of three held-out document folds' strict micro entity F1",
            "grouped_cv": results, "best_params": {"C": best["C"], **lr_options},
            "versions": {name: version(name) for name in ("torch", "transformers", "scikit-learn")},
        }
        return self.report

    def predict(self, tokens: list[str]) -> list[str]:
        self._require_raw()
        if self.classifier is None:
            raise RuntimeError("Train or load lr_raw before prediction")
        if not tokens:
            return []
        features = self.encoder.extract([tokens], batch_size=self.batch_size)
        return _decode(self.classifier.predict(features))

    def save(self, path: str | Path) -> None:
        self._require_raw()
        if self.classifier is None:
            raise RuntimeError("Train or load lr_raw before saving")
        import joblib

        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.encoder.save(target / "encoder")
        joblib.dump(self.classifier, target / "classifier.joblib")
        info = {"model_id": self.model_id, "labels": LABELS, "training_report": self.report}
        (target / "lr_config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FrozenFeatureModel":
        import joblib
        import torch

        target = Path(path)
        info = json.loads((target / "lr_config.json").read_text(encoding="utf-8"))
        if info["model_id"] != "lr_raw" or tuple(info["labels"]) != LABELS:
            raise ValueError("Incompatible lr_raw checkpoint or label order")
        model = cls()
        model.report = info["training_report"]
        model.batch_size = model.report["batch_size"]
        device = model.report["feature_device"]
        try:
            # Checkpoints also load on machines without the training accelerator.
            torch.empty(0, device=device)
        except (AssertionError, RuntimeError):
            device = "cpu"
        model.encoder = FrozenEncoder.load(target / "encoder", device=device)
        model.classifier = joblib.load(target / "classifier.joblib")
        return model
