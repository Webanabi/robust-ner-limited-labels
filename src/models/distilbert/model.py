from __future__ import annotations

from pathlib import Path
from src.models.base import NERModel


class DistilBERTModel(NERModel):
    model_id = "distilbert_clean"

    def train(self, train_rows: list[dict], validation_rows: list[dict], config: dict) -> None:
        # TODO Person 3: implement clean, augmented, ordinary and balanced variants.
        # Align clean/noisy by original word index and score first subwords only.
        # Use overlapping windows for long sentences and deduplicate word predictions.
        raise NotImplementedError("Person 3 must implement DistilBERT training")

    def predict(self, tokens: list[str]) -> list[str]:
        raise NotImplementedError("Person 3 must implement DistilBERT inference")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Person 3 must implement DistilBERT serialization")

    @classmethod
    def load(cls, path: str | Path) -> "DistilBERTModel":
        raise NotImplementedError("Person 3 must implement DistilBERT loading")

