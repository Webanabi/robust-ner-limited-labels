"""Offline CPU regression tests for the frozen DistilBERT + LR baseline."""
from __future__ import annotations

from collections import Counter, defaultdict
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.data.dataset import LABEL_TO_ID, validate_tags
from src.models.registry import create_model, load_model


HAS_LR_DEPS = all(find_spec(name) is not None for name in ("torch", "transformers", "sklearn"))


@unittest.skipUnless(HAS_LR_DEPS, "Person 2 model dependencies are optional")
class FrozenFeatureTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast
        from src.models.lr_pca.encoder import FrozenEncoder

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Alice", "work", "##s",
                 "in", "London", ".", "x"]
        (self.root / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.root, do_lower_case=False, model_max_length=8, local_files_only=True)
        config = DistilBertConfig(vocab_size=len(vocab), max_position_embeddings=8,
                                  dim=12, n_layers=1, n_heads=2, hidden_dim=24)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            self.encoder = FrozenEncoder(self.tokenizer, DistilBertModel(config), device="cpu")
        self.train_config = {"seed": 42, "hyperparameters": {
            "c_values": [0.1, 1.0], "device": "cpu", "batch_size": 2}}

    def test_first_subword_across_windows_batches_and_dropped_words(self):
        import numpy as np
        import torch

        # Echo piece IDs in one coordinate to identify the selected hidden states exactly.
        def mark_pieces(module, args, kwargs, output):
            self.assertFalse(module.training)
            self.assertFalse(torch.is_grad_enabled())
            output.last_hidden_state[:, :, 0] = kwargs["input_ids"]

        handle = self.encoder.encoder.register_forward_hook(mark_pieces, with_kwargs=True)
        self.addCleanup(handle.remove)
        self.assertTrue(all(not p.requires_grad for p in self.encoder.encoder.parameters()))
        words = ["works", "Alice", ".", "in"] * 3
        sentences = [["London"], [], words + ["\x01", "x" + "s" * 15, "London", "works"]]
        expected = ["London"] + ["work", "Alice", ".", "in"] * 3 + ["[UNK]", "x", "London", "work"]
        features = self.encoder.extract(sentences, batch_size=2)
        self.assertEqual(features.shape, (len(expected), 12))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(features[:, 0], self.tokenizer.convert_tokens_to_ids(expected))

    def test_batched_contextual_features_match_reference_encoder(self):
        import numpy as np
        import torch

        sentences = [["works", "in", "London"], ["Alice"]]
        expected = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, is_split_into_words=True, return_tensors="pt")
            with torch.no_grad():
                hidden = self.encoder.encoder(**inputs).last_hidden_state[0]
            word_ids = inputs.word_ids()
            expected.extend(hidden[word_ids.index(i)].numpy() for i in range(len(sentence)))
        actual = self.encoder.extract(sentences, batch_size=2)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_each_candidate_fits_and_scores_disjoint_whole_documents(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from src.models.lr_pca.encoder import FrozenEncoder

        # Two sentences per document, interleaved so sentence-wise folds would leak.
        rows = [{"id": str(i), "doc_id": str(i % 3), "tokens": [str(i % 3)] * 5,
                 "ner_tags": ["B-PER", "O", "O", "O", "O"]} for i in range(6)]
        official = [{"id": "holdout", "tokens": ["OFFICIAL"]}]
        true_fit = LogisticRegression.fit
        fit_docs, held_out = {}, defaultdict(Counter)

        def features(sentences, batch_size):
            # Trying to encode a validation/test word fails immediately.
            return np.array([[int(word), pos] for sent in sentences
                             for pos, word in enumerate(sent)], dtype=np.float32)

        def record_fit(estimator, X, y, *args, **kwargs):
            counts = Counter(X[:, 0].astype(int))
            self.assertEqual(set(counts.values()), {10})
            fit_docs[estimator] = set(counts)
            return true_fit(estimator, X, y, *args, **kwargs)

        def check_prediction(estimator, X):
            documents = X[:, 0].astype(int)
            self.assertTrue(fit_docs[estimator].isdisjoint(documents))
            held_out[estimator.C].update(documents)
            # Predicting all O has higher word accuracy on these sentences.
            # Predicting all B-PER has higher strict entity F1.
            label = "O" if estimator.C == 0.1 else "B-PER"
            return np.full(len(X), LABEL_TO_ID[label])

        self.train_config["validation_variants"] = {"clean": official}
        with patch.object(FrozenEncoder, "pretrained", return_value=self.encoder), \
                patch.object(self.encoder, "extract", side_effect=features), \
                patch.object(LogisticRegression, "fit", record_fit), \
                patch.object(LogisticRegression, "predict", check_prediction):
            report = create_model("lr_raw").train(rows, official, self.train_config)
        self.assertEqual(Counter(map(len, fit_docs.values())), {2: 6, 3: 1})
        self.assertEqual(set(held_out), {0.1, 1.0})
        for counts in held_out.values():
            self.assertEqual(counts, {0: 10, 1: 10, 2: 10})
        self.assertEqual(report["best_params"]["C"], 1.0)
        self.assertAlmostEqual(report["grouped_cv"][1]["mean_f1"], 1 / 3)
        self.assertEqual([len(row["fold_f1"]) for row in report["grouped_cv"]], [3, 3])

    def test_rejects_insufficient_or_unknown_documents(self):
        from src.models.lr_pca.encoder import FrozenEncoder

        for documents in [["d1", "d1", "d2"], ["d1", "d2", "d3", "UNAVAILABLE"]]:
            with self.subTest(documents=documents):
                rows = [{"id": str(i), "doc_id": doc, "tokens": ["."], "ner_tags": ["O"]}
                        for i, doc in enumerate(documents)]
                with patch.object(FrozenEncoder, "pretrained") as download:
                    with self.assertRaisesRegex(ValueError, "document|doc_id"):
                        create_model("lr_raw").train(rows, [], self.train_config)
                    download.assert_not_called()

    def test_registry_checkpoint_round_trip_and_valid_word_predictions(self):
        import numpy as np
        from src.models.lr_pca.encoder import FrozenEncoder

        words = ["Alice", "works", "in", "London", "."]
        rows = [{"id": str(i), "doc_id": str(i), "tokens": words,
                 "ner_tags": ["B-PER", "O", "O", "B-LOC", "O"]} for i in range(3)]
        model = create_model("lr_raw")
        with patch.object(FrozenEncoder, "pretrained", return_value=self.encoder):
            model.train(rows, [], self.train_config)
        checkpoint = self.root / "checkpoint"
        model.save(checkpoint)
        with patch.object(FrozenEncoder, "pretrained", side_effect=AssertionError("remote load")):
            restored = load_model("lr_raw", checkpoint)
        for sentence in [[], ["."], ["works", "unknown"] + words * 4]:
            with self.subTest(sentence=sentence):
                actual = restored.predict_checked(sentence)
                self.assertEqual(actual, model.predict_checked(sentence))
                self.assertEqual(len(actual), len(sentence))
                validate_tags(actual)
        # A word classifier can return an illegal sentence-initial continuation.
        with patch.object(restored.classifier, "predict", return_value=np.array(
                [LABEL_TO_ID["I-LOC"], LABEL_TO_ID["I-LOC"]])):
            self.assertEqual(restored.predict(["in", "London"]), ["B-LOC", "I-LOC"])


if __name__ == "__main__":
    unittest.main()
