"""Deterministic paired perturbations with unchanged word indices."""
from __future__ import annotations

import hashlib
import random
from .dataset import validate_sentence


def _rng(seed: int, sentence_id: str, condition: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{sentence_id}|{condition}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def eligible_typo(word: str) -> bool:
    if len(word) < 4 or not word.isascii() or not word.isalpha():
        return False
    inside = word[1:-1]
    return any(inside[i] != inside[j] for i in range(len(inside)) for j in range(i + 1, len(inside)))


def swap_inside(word: str, rng: random.Random) -> str:
    pairs = [(i, j) for i in range(1, len(word) - 1) for j in range(i + 1, len(word) - 1)
             if word[i] != word[j]]
    if not pairs:
        return word
    i, j = rng.choice(pairs)
    letters = list(word)
    letters[i], letters[j] = letters[j], letters[i]
    return "".join(letters)


def perturb(row: dict, condition: str, seed: int, lower_rate: float = .30,
            typo_rate: float = .10) -> tuple[dict, dict]:
    validate_sentence(row)
    if condition not in ("clean", "lowercase", "typo10", "typo20", "train_lower", "train_typo"):
        raise ValueError(f"Unknown condition {condition}")
    rng = _rng(seed, row["id"], condition)
    tokens = list(row["tokens"])
    eligible = 0
    changed = 0
    for i, word in enumerate(tokens):
        if condition == "clean":
            continue
        if condition == "lowercase":
            candidate, valid, selected = word.lower(), word.lower() != word, True
        elif condition == "train_lower":
            candidate, valid, selected = word.lower(), word.lower() != word, rng.random() < lower_rate
        else:
            valid = eligible_typo(word)
            rate = typo_rate if condition in ("typo10", "train_typo") else .20
            selected = valid and rng.random() < rate
            candidate = swap_inside(word, rng) if selected else word
        eligible += int(valid)
        changed += int(selected and candidate != word)
        tokens[i] = candidate if selected else word
    copy = {**row, "tokens": tokens, "source_id": row["id"], "condition": condition}
    return copy, {"id": row["id"], "condition": condition, "eligible": eligible, "changed": changed}


def training_pair(row: dict, seed: int, lower_rate: float = .30,
                  typo_rate: float = .10, lower_probability: float = .5) -> tuple[dict, dict]:
    if not 0 <= lower_probability <= 1:
        raise ValueError("lower_probability must be in [0, 1]")
    choice = ("train_lower" if _rng(seed, row["id"], "train_choice").random() < lower_probability
              else "train_typo")
    return perturb(row, choice, seed, lower_rate, typo_rate)


def audit_summary(details: list[dict], examples: list[dict] | None = None) -> dict:
    eligible = sum(item["eligible"] for item in details)
    changed = sum(item["changed"] for item in details)
    by_condition: dict[str, int] = {}
    for item in details:
        by_condition[item["condition"]] = by_condition.get(item["condition"], 0) + 1
    return {"sentences": len(details), "eligible_words": eligible,
            "changed_words": changed, "actual_rate": changed / eligible if eligible else 0,
            "conditions": by_condition, "examples": (examples or [])[:8],
            "per_sentence": details}

