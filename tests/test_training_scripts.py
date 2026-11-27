from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_model import run_training
from scripts.test_model import run_testing
from src.data.dataset import write_jsonl
from src.data.sampling import sample_manifest
from src.models.base import NERModel


class FixtureOnlyModel(NERModel):
    model_id = "fixture_only"

    def train(self, train_rows: list[dict], validation_rows: list[dict], config: dict) -> dict:
        assert len(train_rows) == len(config["noisy_train_rows"])
        assert set(config["validation_variants"]) == {"clean", "lowercase", "typo10"}
        self.trained = True
        return {"fixture_only": True}

    def predict(self, tokens: list[str]) -> list[str]:
        if not getattr(self, "trained", False):
            raise RuntimeError("Fixture not trained")
        return ["B-PER" if token.lower() == "alice" else
                "B-LOC" if token.lower() == "london" else "O" for token in tokens]

    def save(self, path: str | Path) -> None:
        (Path(path) / "fixture_weights.json").write_text(
            json.dumps({"trained": self.trained}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FixtureOnlyModel":
        item = json.loads((Path(path) / "fixture_weights.json").read_text(encoding="utf-8"))
        result = cls()
        result.trained = item["trained"]
        return result


class TrainingScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.row = {"id": "train_000000", "doc_id": "train_doc_000001",
                    "tokens": ["Alice", "visits", "London", "."],
                    "ner_tags": ["B-PER", "O", "B-LOC", "O"]}
        write_jsonl(self.data / "train.jsonl", [self.row])
        write_jsonl(self.data / "train_100pct_seed42_noisy.jsonl",
                    [{**self.row, "condition": "train_lower", "source_id": self.row["id"]}])
        (self.data / "train_100pct_seed42_manifest.json").write_text(
            json.dumps(sample_manifest([self.row], 1.0, 42)), encoding="utf-8")
        validation = {**self.row, "id": "validation_000000", "doc_id": "validation_doc_000001"}
        test = {**self.row, "id": "test_000000", "doc_id": "test_doc_000001"}
        write_jsonl(self.data / "validation.jsonl", [validation])
        for condition in ("clean", "lowercase", "typo10", "typo20"):
            write_jsonl(self.data / f"validation_{condition}.jsonl", [validation])
            write_jsonl(self.data / f"test_{condition}.jsonl", [test])
        self.matrix = self.root / "matrix.csv"
        with self.matrix.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "tier", "model_id", "fraction", "seed", "condition", "method",
                "hyperparameters", "status", "checkpoint_path", "prediction_path",
                "metadata_path", "notes"))
            writer.writeheader()
            for condition in ("clean", "lowercase", "typo10", "typo20"):
                writer.writerow({"tier": "fixture", "model_id": "fixture_only",
                                 "fraction": 1.0, "seed": 42, "condition": condition,
                                 "method": "fixture", "hyperparameters": "{}",
                                 "status": "planned"})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_train_then_test_updates_only_measured_rows(self):
        checkpoint_root = self.root / "checkpoints"
        manifest = run_training("fixture_only", 1.0, 42, self.data, checkpoint_root,
                                "fixture_run", {}, {}, self.matrix,
                                model_factory=lambda _: FixtureOnlyModel())
        self.assertEqual(manifest["status"], "trained")
        summary = run_testing("fixture_run", "test", self.data, checkpoint_root,
                              self.root / "predictions", self.root / "metrics",
                              matrix_path=self.matrix,
                              model_loader=lambda _, path: FixtureOnlyModel.load(path))
        self.assertEqual(set(summary["results"]), {"clean", "lowercase", "typo10", "typo20"})
        self.assertEqual(summary["results"]["clean"]["f1"], 1.0)
        self.assertTrue((self.root / "metrics" / "fixture_run" /
                         "test_clean" / "evaluation.json").exists())
        with self.matrix.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["status"] for row in rows}, {"evaluated"})
        self.assertTrue(all(row["prediction_path"] for row in rows))

    def test_unimplemented_model_does_not_claim_training(self):
        root = self.root / "checkpoints"

        class UnimplementedFixture(FixtureOnlyModel):
            def train(self, train_rows, validation_rows, config):
                raise NotImplementedError("fixture training intentionally unavailable")

        with self.assertRaises(NotImplementedError):
            run_training("fixture_only", 1.0, 42, self.data, root, "model_fixture", {}, {},
                         matrix_path=None, model_factory=lambda _: UnimplementedFixture())
        self.assertFalse((root / "model_fixture" / "manifest.json").exists())

    def test_mismatched_subset_stops_before_training(self):
        (self.data / "train_100pct_seed42_manifest.json").write_text(
            json.dumps({**sample_manifest([self.row], 1.0, 42),
                        "sentence_ids": ["wrong"]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sentence IDs differ"):
            run_training("fixture_only", 1.0, 42, self.data, self.root / "checkpoints",
                         "bad_subset", {}, {}, model_factory=lambda _: FixtureOnlyModel())


if __name__ == "__main__":
    unittest.main()

