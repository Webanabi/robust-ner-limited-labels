from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.models.crf.model import CRFModel, sentence_features


class CRFModelTests(unittest.TestCase):
    def test_features_and_save_load(self):
        try:
            import sklearn_crfsuite  # noqa: F401
        except ImportError:
            self.skipTest("sklearn-crfsuite is optional in the public CPU environment")
        tokens = ["Alice", "works", "in", "London", "."]
        standard = sentence_features(tokens, use_case=True, use_affix=True)
        no_case = sentence_features(tokens, use_case=False, use_affix=True)
        no_affix = sentence_features(tokens, use_case=True, use_affix=False)
        self.assertIn("word.istitle", standard[0])
        self.assertNotIn("word.istitle", no_case[0])
        self.assertIn("word.prefix2", standard[0])
        self.assertNotIn("word.prefix2", no_affix[0])
        rows = [
            {"id": f"train_{i:06d}", "doc_id": f"doc_{i}",
             "tokens": tokens, "ner_tags": ["B-PER", "O", "O", "B-LOC", "O"]}
            for i in range(3)
        ]
        model = CRFModel()
        report = model.train(rows, rows, {"hyperparameters": {
            "candidates": [{"c1": 0.0, "c2": 0.1}], "max_iterations": 5}})
        self.assertEqual(len(report["grouped_cv"][0]["fold_f1"]), 3)
        before = model.predict(tokens)
        self.assertEqual(len(before), len(tokens))
        with tempfile.TemporaryDirectory() as temp:
            model.save(temp)
            restored = CRFModel.load(Path(temp))
            self.assertEqual(restored.predict(tokens), before)


if __name__ == "__main__":
    unittest.main()

