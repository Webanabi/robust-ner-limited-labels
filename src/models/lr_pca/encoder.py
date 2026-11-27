"""First-WordPiece features from the frozen, cased DistilBERT base encoder."""
from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

MODEL_NAME = "distilbert/distilbert-base-cased"
MODEL_REVISION = "6ea81172465e8b0ad3fddeed32b986cdcdcffcf0"


class FrozenEncoder:
    def __init__(self, tokenizer, encoder, device="cpu"):
        if not tokenizer.is_fast or encoder.config.model_type != "distilbert":
            raise ValueError("Expected a fast tokenizer and a DistilBERT encoder")
        self.tokenizer = tokenizer
        self.encoder = encoder.to(device).eval().requires_grad_(False)
        self.device = str(device)
        self.max_length = min(tokenizer.model_max_length, encoder.config.max_position_embeddings)
        # DistilBERT wraps a single sequence with [CLS] and [SEP].
        if self.max_length <= 2:
            raise ValueError("The encoder window must have space for at least one wordpiece")

    @classmethod
    def pretrained(cls, device="cpu"):
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
        encoder = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        return cls(tokenizer, encoder, device)

    def save(self, path: str | Path):
        self.tokenizer.save_pretrained(path)
        self.encoder.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path, device="cpu"):
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        encoder = AutoModel.from_pretrained(path, local_files_only=True)
        return cls(tokenizer, encoder, device)

    def _windows(self, tokens, row_offset):
        # Keep global word indices before windowing: even when a word spans two
        # windows, a continuation in the second window cannot become its first piece.
        encoding = self.tokenizer(tokens, is_split_into_words=True, add_special_tokens=False,
                                  truncation=False, verbose=False)
        by_word = [[] for _ in tokens]
        for word, piece in zip(encoding.word_ids(), encoding["input_ids"]):
            by_word[word].append(piece)

        pieces = []
        first_positions = []
        for word_pieces in by_word:
            first_positions.append(len(pieces))
            # A nonempty input word can disappear under BERT's text normalization.
            pieces.extend(word_pieces or [self.tokenizer.unk_token_id])

        capacity = self.max_length - 2
        overlap = min(32, capacity // 4)
        next_word = 0
        for start in range(0, len(pieces), capacity - overlap):
            end = min(start + capacity, len(pieces))
            stop_word = bisect_left(first_positions, end, lo=next_word)
            if stop_word > next_word:
                ids = [self.tokenizer.cls_token_id, *pieces[start:end], self.tokenizer.sep_token_id]
                columns = [1 + position - start for position in first_positions[next_word:stop_word]]
                rows = range(row_offset + next_word, row_offset + stop_word)
                yield ids, rows, columns
                next_word = stop_word
            if next_word == len(tokens):
                break

    def extract(self, sentences: list[list[str]], batch_size: int = 8):
        """Return a contiguous float32 matrix in original sentence and word order."""
        import numpy as np
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        features = np.empty((sum(map(len, sentences)), self.encoder.config.hidden_size),
                            dtype=np.float32)
        batch = []
        row = 0
        with torch.inference_mode():
            for tokens in sentences:
                if tokens:
                    for window in self._windows(tokens, row):
                        batch.append(window)
                        if len(batch) == batch_size:
                            self._write_batch(batch, features)
                            batch.clear()
                row += len(tokens)
            if batch:
                self._write_batch(batch, features)
        return features

    def _write_batch(self, windows, features):
        import torch

        max_len = max(len(ids) for ids, _, _ in windows)
        ids_batch, attention, rows, samples, positions = [], [], [], [], []
        for sample, (ids, word_rows, columns) in enumerate(windows):
            padding = max_len - len(ids)
            ids_batch.append(ids + [self.tokenizer.pad_token_id] * padding)
            attention.append([1] * len(ids) + [0] * padding)
            rows.extend(word_rows)
            samples.extend([sample] * len(columns))
            positions.extend(columns)

        hidden = self.encoder(input_ids=torch.tensor(ids_batch, device=self.device),
                              attention_mask=torch.tensor(attention, device=self.device)).last_hidden_state
        features[rows] = hidden[samples, positions].float().cpu().numpy()
