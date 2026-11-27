"""Word-level DistilBERT token classification and consistency training."""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src.data.dataset import ID_TO_LABEL, LABEL_TO_ID, repair_iob2
from src.evaluation.metrics import score
from src.models.base import NERModel

CHECKPOINT = "distilbert/distilbert-base-cased"
METHODS = ("distilbert_clean", "distilbert_aug", "distilbert_consistency", "distilbert_balanced")
VALIDATION = ("clean", "lowercase", "typo10")
O_ID = LABEL_TO_ID["O"]


def word_jsd(clean_logits, noisy_logits):
    import torch
    from torch.nn.functional import log_softmax

    p = log_softmax(clean_logits.float(), dim=-1)
    q = log_softmax(noisy_logits.float(), dim=-1)
    midpoint = torch.logaddexp(p, q) - math.log(2)
    divergence = (p.exp() * (p - midpoint) + q.exp() * (q - midpoint)).sum(dim=-1)
    return divergence.mul(0.5).clamp_min(0)


def balanced_jsd(divergence, labels, group_counts=None):
    """Each present gold group contributes equally over an optimizer batch."""
    entity = labels != O_ID
    if group_counts is None:
        group_counts = (int(entity.sum()), int((~entity).sum()))
    terms = [divergence[mask].sum() / count
             for mask, count in zip((entity, ~entity), group_counts) if count]
    return sum(terms) / len(terms)


@dataclass
class Encoding:
    windows: list[list[int]]
    owners: list[tuple[int, int]]


def encode_words(tokenizer, words, max_length, overlap):
    """Pack whole words into windows and assign each word to its most central copy."""
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_length - special_count
    if budget < 1 or not 0 <= overlap < budget:
        raise ValueError("overlap must be smaller than the available WordPiece budget")
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False,
                        truncation=False, verbose=False)
    groups = [[] for _ in words]
    for word, piece in zip(encoded.word_ids(), encoded["input_ids"]):
        groups[word].append(piece)
    # Preserve even empty-normalized words and represent an overlong word by its prefix.
    groups = [(pieces or [tokenizer.unk_token_id])[:budget] for pieces in groups]
    windows = []
    owners = [None] * len(words)
    margins = [-1] * len(words)
    left = 0
    while left < len(groups):
        right = left
        payload = []
        starts = []
        while right < len(groups) and len(payload) + len(groups[right]) <= budget:
            starts.append(len(payload))
            payload.extend(groups[right])
            right += 1
        # The selected DistilBERT tokenizer wraps one sequence with CLS and SEP.
        ids = [tokenizer.cls_token_id, *payload, tokenizer.sep_token_id]
        if special_count != 2:
            raise ValueError("Expected a DistilBERT tokenizer with CLS and SEP")
        for word, position in enumerate(starts, left):
            margin = min(position, len(payload) - position - len(groups[word]))
            if margin > margins[word]:
                margins[word] = margin
                owners[word] = (len(windows), position + 1)
        windows.append(ids)
        if right == len(groups):
            break
        next_left = right
        retained = 0
        while next_left > left + 1 and retained < overlap:
            next_left -= 1
            retained += len(groups[next_left])
        # Ensure the new window can add at least the next previously unseen word.
        while retained + len(groups[right]) > budget:
            retained -= len(groups[next_left])
            next_left += 1
        left = next_left
    return Encoding(windows, owners)


class DistilBERTModel(NERModel):
    model_id = "distilbert_clean"

    def _initialize(self, seed):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=self.revision, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            CHECKPOINT, revision=self.revision, num_labels=len(LABEL_TO_ID),
            id2label=ID_TO_LABEL, label2id=LABEL_TO_ID,
        ).to(self.device)
        if self.settings["max_length"] > self.model.config.max_position_embeddings:
            raise ValueError("max_length exceeds the model's positional capacity")

    def _encode(self, words):
        return encode_words(self.tokenizer, words, self.settings["max_length"], self.settings["overlap"])

    def _word_logits(self, encodings):
        import torch

        windows = []
        owners = []
        for encoding in encodings:
            owners.extend((len(windows) + win, pos) for win, pos in encoding.owners)
            windows.extend(encoding.windows)
        batch = self.tokenizer.pad({"input_ids": windows}, padding=True, return_tensors="pt")
        logits = self.model(**batch.to(self.device)).logits
        window_ids, positions = zip(*owners)
        return logits[torch.tensor(window_ids, device=self.device),
                      torch.tensor(positions, device=self.device)]

    def _validation_score(self, variants):
        scores = {}
        for condition, items in variants.items():
            rows, encodings = items
            tags = self._predict_encoded(encodings)
            guessed = [dict(id=row["id"], tokens=row["tokens"], pred_tags=predicted)
                       for row, predicted in zip(rows, tags)]
            scores[condition] = score(rows, guessed)["micro"]["f1"]
        return scores

    def _predict_encoded(self, encodings):
        import torch

        self.model.eval()
        predictions = []
        batch_size = self.settings["eval_batch_size"]
        with torch.inference_mode():
            for start in range(0, len(encodings), batch_size):
                batch = encodings[start:start + batch_size]
                ids = self._word_logits(batch).argmax(dim=-1).tolist()
                offset = 0
                for encoding in batch:
                    length = len(encoding.owners)
                    predictions.append(repair_iob2([ID_TO_LABEL[i] for i in ids[offset:offset + length]]))
                    offset += length
        return predictions

    def _train_epoch(self, examples, order, optimizer, scaler, precision, coefficient):
        import torch
        from torch.nn.functional import cross_entropy

        self.model.train()
        micro_batch = self.settings["micro_batch"]
        effective_batch = micro_batch * self.settings["gradient_accumulation"]
        losses = []
        for start in range(0, len(order), effective_batch):
            logical = [examples[i] for i in order[start:start + effective_batch]]
            total_words = sum(len(item[2]) for item in logical)
            o_count = sum(item[2].count(O_ID) for item in logical)
            group_counts = (total_words - o_count, o_count)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=self.device)
            for offset in range(0, len(logical), micro_batch):
                clean, noisy, tags = zip(*logical[offset:offset + micro_batch])
                labels = torch.tensor([tag for row in tags for tag in row], device=self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=precision):
                    original = self._word_logits(clean)
                    loss = cross_entropy(original.float(), labels, reduction="sum") / total_words
                    if self.model_id != "distilbert_clean":
                        perturbed = self._word_logits(noisy)
                        noisy_ce = cross_entropy(perturbed.float(), labels, reduction="sum") / total_words
                        loss = (loss + noisy_ce) / 2
                        if coefficient:
                            divergence = word_jsd(original, perturbed)
                            jsd = (balanced_jsd(divergence, labels, group_counts)
                                   if self.model_id == "distilbert_balanced"
                                   else divergence.sum() / total_words)
                            loss = loss + coefficient * jsd
                scaler.scale(loss).backward()
                batch_loss += loss.detach()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(batch_loss)
        return torch.stack(losses).mean().item()

    def train(self, train_rows, validation_rows, config):
        import torch

        started = time.perf_counter()
        self.model_id = config.get("model_id", self.model_id)
        if self.model_id not in METHODS:
            raise ValueError(f"Unknown DistilBERT method: {self.model_id}")
        hp = config.get("hyperparameters", {})
        self.settings = dict(epochs=3, learning_rate=5e-5, weight_decay=0.01,
                             micro_batch=1, gradient_accumulation=8, max_length=256,
                             overlap=32, eval_batch_size=8, amp=True)
        for key in self.settings:
            if key in hp:
                self.settings[key] = hp[key]
        for key in ("epochs", "micro_batch", "gradient_accumulation", "max_length", "eval_batch_size"):
            if not isinstance(self.settings[key], int) or self.settings[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        self.device = torch.device(hp.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.revision = hp.get("revision", "main")
        seed = config.get("seed", 42)
        validation = config["validation_variants"]
        if not train_rows or any(not row["tokens"] for row in train_rows):
            raise ValueError("Training requires nonempty sentences")
        if any(not validation.get(condition) for condition in VALIDATION):
            raise ValueError("Clean, Lowercase and Typo-10 validation variants are required")
        if self.model_id == "distilbert_clean":
            noisy_rows = [None] * len(train_rows)
        else:
            noisy_rows = config["noisy_train_rows"]
            if len(noisy_rows) != len(train_rows) or any(
                clean["id"] != noisy["id"] or len(clean["tokens"]) != len(noisy["tokens"])
                or clean["ner_tags"] != noisy["ner_tags"] for clean, noisy in zip(train_rows, noisy_rows)
            ):
                raise ValueError("Clean and noisy training examples must match by sentence and original word index")

        self._initialize(seed)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        examples = [
            (self._encode(row["tokens"]), self._encode(noisy["tokens"]) if noisy else None,
             [LABEL_TO_ID[tag] for tag in row["ner_tags"]])
            for row, noisy in zip(train_rows, noisy_rows)
        ]
        validation_inputs = {
            condition: (validation[condition], [self._encode(row["tokens"]) for row in validation[condition]])
            for condition in VALIDATION
        }
        candidates = (0.1, 0.5, 1.0) if self.model_id in METHODS[2:] else (0.0,)
        precision = bool(self.settings["amp"] and self.device.type == "cuda")
        best = None
        best_state = None
        history = []
        for candidate_index, coefficient in enumerate(candidates):
            if candidate_index:
                del optimizer, scaler, self.model
                self._initialize(seed)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.settings["learning_rate"],
                                          weight_decay=self.settings["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=precision) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=precision)
            rng = random.Random(seed)
            for epoch in range(1, self.settings["epochs"] + 1):
                order = list(range(len(examples)))
                rng.shuffle(order)
                loss = self._train_epoch(examples, order, optimizer, scaler, precision, coefficient)
                condition_scores = self._validation_score(validation_inputs)
                mean_f1 = sum(condition_scores.values()) / len(VALIDATION)
                record = dict(epoch=epoch, lambda_value=coefficient, training_loss=loss,
                              validation_f1=condition_scores, mean_validation_f1=mean_f1)
                history.append(record)
                print(f"{self.model_id} lambda={coefficient:g} epoch={epoch} loss={loss:.5f} val_f1={mean_f1:.5f}", flush=True)
                if best is None or mean_f1 > best["mean_validation_f1"]:
                    best = record
                    best_state = {key: value.detach().to("cpu", copy=True)
                                  for key, value in self.model.state_dict().items()}
        del optimizer, scaler
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.training_report = dict(
            best=best, history=history, settings=self.settings, seed=seed, checkpoint=CHECKPOINT,
            revision=getattr(self.model.config, "_commit_hash", None) or self.revision,
            effective_sentence_batch=self.settings["micro_batch"] * self.settings["gradient_accumulation"],
            planned_updates_per_candidate=self.settings["epochs"] * math.ceil(
                len(train_rows) / (self.settings["micro_batch"] * self.settings["gradient_accumulation"])),
            candidate_count=len(candidates), actual_amp=precision, device=str(self.device),
            gpu_name=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
            torch_version=str(torch.__version__), training_seconds=time.perf_counter() - started,
        )
        return self.training_report

    def predict(self, tokens):
        if not tokens:
            return []
        return self._predict_encoded([self._encode(tokens)])[0]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        (path / "distilbert_run.json").write_text(json.dumps({
            "model_id": self.model_id, "training_report": self.training_report,
        }, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        path = Path(path)
        record = json.loads((path / "distilbert_run.json").read_text(encoding="utf-8"))
        result = cls()
        result.model_id = record["model_id"]
        result.training_report = record["training_report"]
        result.settings = result.training_report["settings"]
        result.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        result.model = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True).to(result.device).eval()
        if result.model.config.id2label != ID_TO_LABEL:
            raise ValueError("Saved model does not use the repository's IOB2 label map")
        return result
