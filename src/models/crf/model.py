"""CRF baseline using sklearn-crfsuite, with document-grouped model selection."""
from __future__ import annotations

import json
from pathlib import Path

from src.data.dataset import repair_iob2
from src.evaluation.metrics import score
from src.models.base import NERModel


def word_features(tokens: list[str], index: int, *, use_case: bool,
                  use_affix: bool) -> dict:
    word = tokens[index]
    lower = word.lower()
    features = {
        "bias": 1.0,
        "word.lower": lower,
        "word.isdigit": word.isdigit(),
        "word.has_digit": any(char.isdigit() for char in word),
        "word.length": min(len(word), 20),
    }
    if use_case:
        features.update({
            "word.isupper": word.isupper(),
            "word.istitle": word.istitle(),
            "word.islower": word.islower(),
        })
    if use_affix:
        for length in (2, 3, 4):
            features[f"word.prefix{length}"] = lower[:length]
            features[f"word.suffix{length}"] = lower[-length:]
    if index == 0:
        features["BOS"] = True
    else:
        previous = tokens[index - 1]
        features["prev.lower"] = previous.lower()
        if use_case:
            features["prev.istitle"] = previous.istitle()
    if index == len(tokens) - 1:
        features["EOS"] = True
    else:
        following = tokens[index + 1]
        features["next.lower"] = following.lower()
        if use_case:
            features["next.istitle"] = following.istitle()
    return features


def sentence_features(tokens: list[str], *, use_case: bool,
                      use_affix: bool) -> list[dict]:
    return [word_features(tokens, i, use_case=use_case, use_affix=use_affix)
            for i in range(len(tokens))]


class CRFModel(NERModel):
    model_id = "crf"

    def __init__(self) -> None:
        self.model = None
        self.use_case = True
        self.use_affix = True
        self.best_params: dict = {}
        self.cv_results: list[dict] = []

    def train(self, train_rows: list[dict], validation_rows: list[dict],
              config: dict) -> dict:
        try:
            import sklearn_crfsuite
            from sklearn.model_selection import GroupKFold
        except ImportError as exc:
            raise RuntimeError("Install sklearn-crfsuite in the training environment") from exc
        if self.model_id not in ("crf", "crf_no_case", "crf_no_affix"):
            raise ValueError(f"Unsupported CRF variant: {self.model_id}")
        self.use_case = self.model_id != "crf_no_case"
        self.use_affix = self.model_id != "crf_no_affix"
        groups = [row["doc_id"] for row in train_rows]
        if "UNAVAILABLE" in groups or len(set(groups)) < 3:
            raise ValueError("CRF grouped CV needs at least three real documents")
        hyper = config.get("hyperparameters", {})
        candidates = hyper.get("candidates", [
            {"c1": 0.0, "c2": 0.1},
            {"c1": 0.1, "c2": 0.1},
            {"c1": 0.1, "c2": 1.0},
        ])
        if not candidates or not all("c1" in item and "c2" in item for item in candidates):
            raise ValueError("CRF candidates must contain c1 and c2")
        max_iterations = int(hyper.get("max_iterations", 75))
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        X = [sentence_features(row["tokens"], use_case=self.use_case,
                               use_affix=self.use_affix) for row in train_rows]
        y = [row["ner_tags"] for row in train_rows]
        folds = list(GroupKFold(n_splits=3).split(X, y, groups))
        for train_index, valid_index in folds:
            if {groups[i] for i in train_index} & {groups[i] for i in valid_index}:
                raise AssertionError("Document leakage in grouped CV")
        results = []
        for candidate in candidates:
            fold_scores = []
            for fold_number, (train_index, valid_index) in enumerate(folds, 1):
                estimator = sklearn_crfsuite.CRF(
                    algorithm="lbfgs", c1=float(candidate["c1"]),
                    c2=float(candidate["c2"]), max_iterations=max_iterations,
                    all_possible_transitions=True,
                )
                estimator.fit([X[i] for i in train_index], [y[i] for i in train_index])
                predicted = estimator.predict([X[i] for i in valid_index])
                gold_rows = [train_rows[i] for i in valid_index]
                pred_rows = [{"id": row["id"], "tokens": row["tokens"],
                              "pred_tags": repair_iob2(list(tags))}
                             for row, tags in zip(gold_rows, predicted)]
                f1 = score(gold_rows, pred_rows)["micro"]["f1"]
                fold_scores.append(f1)
                print(f"CRF CV c1={candidate['c1']} c2={candidate['c2']} "
                      f"fold={fold_number} F1={f1:.4f}", flush=True)
            results.append({"c1": float(candidate["c1"]), "c2": float(candidate["c2"]),
                            "fold_f1": fold_scores,
                            "mean_f1": sum(fold_scores) / len(fold_scores)})
        best = max(results, key=lambda item: item["mean_f1"])
        self.best_params = {"c1": best["c1"], "c2": best["c2"],
                            "max_iterations": max_iterations}
        self.cv_results = results
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs", **self.best_params, all_possible_transitions=True,
        )
        self.model.fit(X, y)
        return {"best_params": self.best_params, "grouped_cv": results,
                "training_documents": len(set(groups)),
                "feature_variant": self.model_id}

    def predict(self, tokens: list[str]) -> list[str]:
        if self.model is None:
            raise RuntimeError("CRF model is not trained")
        features = sentence_features(tokens, use_case=self.use_case,
                                     use_affix=self.use_affix)
        return repair_iob2(list(self.model.predict_single(features)))

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("CRF model is not trained")
        import joblib
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, target / "model.joblib")
        (target / "crf_config.json").write_text(
            json.dumps({"model_id": self.model_id, "use_case": self.use_case,
                        "use_affix": self.use_affix, "best_params": self.best_params,
                        "cv_results": self.cv_results}, indent=2),
            encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CRFModel":
        import joblib
        target = Path(path)
        info = json.loads((target / "crf_config.json").read_text(encoding="utf-8"))
        result = cls()
        result.model_id = info["model_id"]
        result.use_case = info["use_case"]
        result.use_affix = info["use_affix"]
        result.best_params = info["best_params"]
        result.cv_results = info["cv_results"]
        result.model = joblib.load(target / "model.joblib")
        return result

