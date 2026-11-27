import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.dataset import ID_TO_LABEL, LABEL_TO_ID, validate_tags
from src.models.distilbert.model import METHODS, DistilBERTModel, balanced_jsd, encode_words, word_jsd


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ("torch", "transformers")),
                     "DistilBERT tests require torch and transformers")
class DistilBERTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import BertWordPieceTokenizer
        from transformers import BertTokenizerFast

        cls.torch = torch
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.addClassCleanup(torch.set_num_threads, cls.original_threads)
        root = Path(cls.scratch.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                 "New", "new", "York", "y", "##ork", "visits", "."]
        vocab_path = root / "vocab.txt"
        vocab_path.write_text("\n".join(vocab), encoding="utf-8")
        tokenizer_path = root / "tokenizer.json"
        BertWordPieceTokenizer(str(vocab_path), lowercase=False).save(str(tokenizer_path))
        cls.tokenizer = BertTokenizerFast(tokenizer_file=str(tokenizer_path), do_lower_case=False)

    def test_jsd_is_stable_and_differentiable_through_both_views(self):
        torch = self.torch
        a = torch.tensor([[10000., -10000.], [2., -1.], [1., 0.]], requires_grad=True)
        b = torch.tensor([[-10000., 10000.], [0., 1.], [-2., 3.]], requires_grad=True)
        values = word_jsd(a, b)
        torch.testing.assert_close(values[0], torch.tensor(0.6931472))
        torch.testing.assert_close(word_jsd(a, a), torch.zeros(3), atol=1e-7, rtol=0)
        values.sum().backward()
        for gradient in (a.grad, b.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_group_balance_and_accumulation_match(self):
        torch = self.torch
        values = torch.tensor([.1, .2, .6])
        for tags, expected in (([0, 0, 0], .3), ([1, 1, 1], .3), ([0, 0, 1], .375)):
            labels = torch.tensor(tags)
            torch.testing.assert_close(balanced_jsd(values, labels), torch.tensor(expected))
        labels = torch.tensor([0, 0, 1])
        combined = sum(balanced_jsd(v, y, (1, 2)) for v, y in zip(values.split(2), labels.split(2)))
        torch.testing.assert_close(combined, balanced_jsd(values, labels))

    def test_wordpiece_alignment_and_window_ownership(self):
        t = self.tokenizer
        clean = ["New", "York", "visits", "."] * 4
        noisy = ["new", "york", "visits", "."] * 4
        def first_pieces(words):
            encoding = encode_words(t, words, max_length=7, overlap=2)
            self.assertGreater(len(encoding.windows), 1)
            self.assertTrue(all(len(window) <= 7 for window in encoding.windows))
            self.assertEqual(len(encoding.owners), len(words))
            self.assertEqual(len(set(encoding.owners)), len(words))
            pieces = [encoding.windows[window][offset] for window, offset in encoding.owners]
            return t.convert_ids_to_tokens(pieces)
        self.assertEqual(first_pieces(clean), clean)
        self.assertEqual(first_pieces(noisy), ["new", "y", "visits", "."] * 4)
        self.assertEqual(t(noisy[:2], is_split_into_words=True, add_special_tokens=False).word_ids(), [0, 1, 1])

    def test_all_objectives_train_select_predict_and_roundtrip(self):
        from transformers import DistilBertConfig, DistilBertForTokenClassification

        torch = self.torch
        tokenizer = self.tokenizer

        def tiny_model(model, seed):
            torch.manual_seed(seed)
            model.tokenizer = tokenizer
            model.model = DistilBertForTokenClassification(DistilBertConfig(
                vocab_size=len(tokenizer), dim=16, hidden_dim=24, n_layers=1, n_heads=2,
                max_position_embeddings=16, label2id=LABEL_TO_ID, id2label=ID_TO_LABEL,
            )).to(model.device)

        rows = [
            dict(id="1", tokens=["New", "York", "visits", "."], ner_tags=["B-LOC", "I-LOC", "O", "O"]),
            dict(id="2", tokens=["visits", "."], ner_tags=["O", "O"]),
        ]
        noisy = [dict(row, tokens=[word.lower() for word in row["tokens"]]) for row in rows]
        validation = [dict(row, id="validation_" + row["id"]) for row in rows]
        noisy_validation = [dict(row, tokens=[word.lower() for word in row["tokens"]]) for row in validation]
        with patch.object(DistilBERTModel, "_initialize", tiny_model):
            for method in METHODS:
                with self.subTest(method=method):
                    model = DistilBERTModel()
                    device = os.environ.get("PERSON3_TEST_DEVICE", "cpu")
                    report = model.train(rows, validation, {
                        "model_id": method, "seed": 42, "noisy_train_rows": noisy,
                        "validation_variants": {"clean": validation, "lowercase": noisy_validation,
                                                "typo10": noisy_validation},
                        "hyperparameters": {"device": device, "epochs": 1, "max_length": 7,
                                            "overlap": 2, "gradient_accumulation": 2},
                    })
                    candidates = [.1, .5, 1.] if method in METHODS[2:] else [0.]
                    self.assertEqual([r["lambda_value"] for r in report["history"]], candidates)
                    self.assertEqual(set(report["best"]["validation_f1"]), {"clean", "lowercase", "typo10"})
                    self.assertEqual(report["best"]["mean_validation_f1"],
                                     max(r["mean_validation_f1"] for r in report["history"]))
                    self.assertEqual(report["planned_updates_per_candidate"], 1)
                    sentence = rows[0]["tokens"] * 4
                    tags = model.predict_checked(sentence)
                    validate_tags(tags)
                    with tempfile.TemporaryDirectory() as directory:
                        model.save(directory)
                        restored = DistilBERTModel.load(directory)
                        restored.model.to(model.device)
                        restored.device = model.device
                        self.assertEqual(restored.predict_checked(sentence), tags)
                        self.assertEqual(restored.model_id, method)


if __name__ == "__main__":
    unittest.main()
