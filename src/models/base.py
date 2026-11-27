"""Contract implemented by every model owner."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from src.data.dataset import validate_tags


class NERModel(ABC):
    model_id: str

    @abstractmethod
    def train(self, train_rows: list[dict], validation_rows: list[dict],
              config: dict) -> dict | None:
        """Fit only on train_rows; optionally return JSON-ready training details."""

    @abstractmethod
    def predict(self, tokens: list[str]) -> list[str]:
        """One word-level IOB2 tag per original token."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save weights plus parameters needed by load."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "NERModel":
        """Restore a trained model, never a fabricated fallback."""

    def predict_checked(self, tokens: list[str]) -> list[str]:
        tags = self.predict(tokens)
        if len(tags) != len(tokens):
            raise ValueError(f"{self.model_id}: returned {len(tags)} labels for {len(tokens)} words")
        validate_tags(tags, strict_bio=False)
        return tags

