from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.data.dataset import LABEL_TO_ID, read_jsonl, repair_iob2, spans, write_jsonl
from src.data.noise import eligible_typo, perturb, training_pair
from src.data.preprocess import read_conll
from src.data.sampling import sample_manifest, sample_sentences
from src.evaluation.evaluate import evaluate_files
from src.evaluation.error_analysis import analyze
from src.evaluation.metrics import score
from src.analysis.aggregate import aggregate
from src.analysis.visualization import generate
from src.models.base import NERModel
from src.models.predict import export_predictions
from src.demo import tokenize_with_offsets, entities, highlight


def sentence(index: int, tags: list[str] | None = None) -> dict:
    return {"id": f"test_{index:06d}", "doc_id": f"doc_{index // 2}",
            "tokens": ["Alice", "works", "in", "London", "."],
            "ner_tags": tags or ["B-PER", "O", "O", "B-LOC", "O"]}


class PublicTests(unittest.TestCase):
    def test_demo_offsets_and_safe_markup(self):
        text = "Hi, Alice at <Acme>!"
        tokens, offsets = tokenize_with_offsets(text)
        tags = ["O", "O", "B-PER", "O", "O", "B-ORG", "O", "O"]
        found = entities(tokens, offsets, tags, text)
        self.assertEqual(found[0], {"text": "Alice", "type": "PER", "start": 4, "end": 9})
        self.assertEqual(found[1]["text"], "Acme")
        self.assertIn("&lt;", highlight(text, found))

    def test_prediction_export_contract(self):
        class FixtureModel(NERModel):
            model_id = "fixture_only"

            def train(self, train_rows, validation_rows, config):
                raise NotImplementedError

            def predict(self, tokens):
                return ["O"] * len(tokens)

            def save(self, path):
                raise NotImplementedError

            @classmethod
            def load(cls, path):
                raise NotImplementedError

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.jsonl"
            meta = {"model_id": "fixture_only", "fraction": .1, "seed": 42,
                    "condition": "clean", "training_seconds": 0,
                    "hardware": "CPU test fixture", "hyperparameters": {}}
            report = export_predictions(FixtureModel(), [sentence(1)], path, meta)
            self.assertEqual(report["sentences"], 1)
            self.assertEqual(read_jsonl(path, "prediction")[0]["pred_tags"], ["O"] * 5)
            self.assertTrue(path.with_name("fixture_metadata.json").exists())

    def test_prepare_data_cli_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            content = ("-DOCSTART- -X- -X- O\n\n"
                       "Alice NNP B-NP B-PER\n"
                       "works VBZ B-VP O\n\n"
                       "-DOCSTART- -X- -X- O\n\n"
                       "London NNP B-NP B-LOC\n\n")
            paths = []
            for name in ("train", "validation", "test"):
                path = root / f"{name}.txt"
                path.write_text(content, encoding="utf-8")
                paths.append(path)
            output = root / "out"
            command = [sys.executable, "-m", "scripts.prepare_data", "--source", "local",
                       "--train", str(paths[0]), "--validation", str(paths[1]),
                       "--test", str(paths[2]), "--output", str(output)]
            run = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(len(read_jsonl(output / "test_typo10.jsonl")), 2)
            self.assertTrue((output / "train_10pct_seed42_manifest.json").exists())
            audit = json.loads((output / "test_typo10_noise_audit.json").read_text())
            self.assertIn("actual_rate", audit)

    def test_labels_and_iob2(self):
        self.assertEqual(len(LABEL_TO_ID), 9)
        self.assertEqual(repair_iob2(["I-PER", "I-PER", "O", "I-ORG"]),
                         ["B-PER", "I-PER", "O", "B-ORG"])
        self.assertEqual(spans(["B-PER", "I-PER", "O", "B-LOC"]),
                         [(0, 2, "PER"), (3, 4, "LOC")])

    def test_local_conll_keeps_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.txt"
            path.write_text("-DOCSTART- -X- -X- O\n\nAlice NNP B-NP B-PER\n\n"
                            "-DOCSTART- -X- -X- O\n\nLondon NNP B-NP B-LOC\n",
                            encoding="utf-8")
            rows = read_conll(path, "train")
            self.assertEqual([r["doc_id"] for r in rows],
                             ["train_doc_000001", "train_doc_000002"])
            self.assertEqual(rows[1]["ner_tags"], ["B-LOC"])

    def test_original_iob1_converts_to_iob2(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "original.txt"
            path.write_text("-DOCSTART- -X- -X- O\n\n"
                            "Alice NNP B-NP I-PER\n"
                            "met VBD B-VP O\n"
                            "Bob NNP B-NP I-PER\n", encoding="utf-8")
            rows = read_conll(path, "train")
            self.assertEqual(rows[0]["ner_tags"], ["B-PER", "O", "B-PER"])
            self.assertEqual(rows[0]["source_tag_conversions"], 2)

    def test_sampling_deterministic(self):
        rows = [sentence(i) for i in range(100)]
        one = sample_sentences(rows, .1, 42)
        two = sample_sentences(rows, .1, 42)
        self.assertEqual([r["id"] for r in one], [r["id"] for r in two])
        self.assertEqual(len(one), 10)
        manifest = sample_manifest(one, .1, 42)
        self.assertEqual(manifest["word_count"], 50)
        self.assertEqual(manifest["entity_count"], 20)

    def test_noise_preserves_alignment_and_is_order_independent(self):
        row = sentence(1)
        variant, audit = training_pair(row, 42)
        self.assertEqual(len(variant["tokens"]), len(row["tokens"]))
        self.assertEqual(variant["ner_tags"], row["ner_tags"])
        self.assertEqual(variant, training_pair(row, 42)[0])
        self.assertEqual(perturb(row, "typo20", 9), perturb(row, "typo20", 9))
        self.assertFalse(eligible_typo("aaaa"))
        self.assertTrue(eligible_typo("London"))
        self.assertLessEqual(audit["changed"], audit["eligible"])
        self.assertEqual(row["tokens"][0], "Alice")

    def test_exact_metrics_and_error_categories(self):
        gold = [sentence(1)]
        pred = [{"id": gold[0]["id"], "tokens": gold[0]["tokens"],
                 "pred_tags": ["B-PER", "O", "O", "B-ORG", "O"]}]
        metrics = score(gold, pred)
        self.assertEqual(metrics["micro"]["correct"], 1)
        self.assertAlmostEqual(metrics["micro"]["f1"], .5)
        errors = analyze(gold, pred)
        self.assertEqual(errors["counts"], {"wrong_type": 1})
        pred[0]["pred_tags"] = ["O", "O", "B-LOC", "I-LOC", "O"]
        self.assertEqual(analyze(gold, pred)["counts"],
                         {"boundary": 1, "missed": 1})

    def test_evaluation_validation_and_report_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold_path, pred_path = root / "gold.jsonl", root / "pred.jsonl"
            gold = [sentence(1), sentence(2)]
            pred = [{"id": row["id"], "tokens": row["tokens"],
                     "pred_tags": row["ner_tags"]} for row in reversed(gold)]
            write_jsonl(gold_path, gold)
            write_jsonl(pred_path, pred, "prediction")
            self.assertEqual(read_jsonl(pred_path, "prediction"), pred)
            meta = {"model_id": "test_fixture", "fraction": .1, "seed": 42,
                    "condition": "clean", "training_seconds": 1.0,
                    "inference_seconds": .1, "hardware": "CPU fixture",
                    "hyperparameters": {}}
            meta_path = root / "metadata.json"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            result = evaluate_files(gold_path, pred_path, root / "evaluation", meta_path)
            self.assertEqual(result["metrics"]["micro"]["f1"], 1)
            aggregate([root / "evaluation" / "evaluation.json"], root / "tables")
            figures = generate(root / "tables", root / "figures")
            self.assertTrue(figures)
            self.assertTrue((root / "tables" / "summary.csv").exists())
            self.assertTrue((root / "evaluation" / "errors.csv").exists())
            pred.pop()
            write_jsonl(pred_path, pred, "prediction")
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                evaluate_files(gold_path, pred_path, root / "bad")


if __name__ == "__main__":
    unittest.main()

