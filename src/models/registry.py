from __future__ import annotations

from pathlib import Path
from .crf.model import CRFModel
from .lr_pca.model import FrozenFeatureModel
from .distilbert.model import DistilBERTModel

MODEL_CLASSES = {
    "crf": CRFModel, "crf_no_case": CRFModel, "crf_no_affix": CRFModel,
    "lr_raw": FrozenFeatureModel, "lr_pca64": FrozenFeatureModel,
    "lr_pca128": FrozenFeatureModel, "distilbert_clean": DistilBERTModel,
    "distilbert_aug": DistilBERTModel,
    "distilbert_consistency": DistilBERTModel,
    "distilbert_balanced": DistilBERTModel,
}


def create_model(model_id: str):
    if model_id not in MODEL_CLASSES:
        raise ValueError(f"Unknown model_id: {model_id}")
    model = MODEL_CLASSES[model_id]()
    model.model_id = model_id
    return model


def load_model(model_id: str, path: str | Path):
    if model_id not in MODEL_CLASSES:
        raise ValueError(f"Unknown model_id: {model_id}")
    model = MODEL_CLASSES[model_id].load(path)
    model.model_id = model_id
    return model

