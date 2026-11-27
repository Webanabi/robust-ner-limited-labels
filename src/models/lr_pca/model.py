from __future__ import annotations

from pathlib import Path
from src.models.base import NERModel


class FrozenFeatureModel(NERModel):
    model_id = "lr_raw"

    def train(self, train_rows: list[dict], validation_rows: list[dict], config: dict) -> None:
        # TODO Person 2: first-subword frozen DistilBERT vectors, LR, optional PCA 64/128.
        # Refit PCA within each grouped CV training fold; never fit it on validation/test.
        raise NotImplementedError("Person 2 must implement frozen features and LR")

    def predict(self, tokens: list[str]) -> list[str]:
        raise NotImplementedError("Person 2 must implement frozen feature inference")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Person 2 must save encoder reference, PCA and LR")

    @classmethod
    def load(cls, path: str | Path) -> "FrozenFeatureModel":
        raise NotImplementedError("Person 2 must implement frozen feature loading")

