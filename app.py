"""Streamlit shell. Inference becomes available when teammates implement load/predict."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import streamlit as st
from src.data.noise import perturb
from src.demo import entities, highlight, tokenize_with_offsets
from src.models.registry import load_model


st.title("NER demo")
st.caption("Only trained checkpoints are shown. CRF is available; LR/PCA and DistilBERT remain TODO.")
root = Path("checkpoints")
manifests = sorted(root.glob("*/manifest.json")) if root.exists() else []
choices = {}
for path in manifests:
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") == "trained" and item.get("model_id"):
            choices[path.parent.name] = (item["model_id"], path.parent)
    except (ValueError, OSError):
        pass
if not choices:
    st.info("No trained model manifest found in checkpoints/<run>/manifest.json.")
    st.stop()
selected = st.selectbox("Model", list(choices))
sentence = st.text_input("English sentence", "I study at City University of Hong Kong.")
condition = st.selectbox("Compare with", ["lowercase", "typo10", "typo20"])
if st.button("Recognize") and sentence.strip():
    model_id, path = choices[selected]
    try:
        model = load_model(model_id, path)
        tokens, offsets = tokenize_with_offsets(sentence)
        clean_tags = model.predict_checked(tokens)
        clean = entities(tokens, offsets, clean_tags, sentence)
        row = {"id": "demo", "doc_id": "demo", "tokens": tokens, "ner_tags": ["O"] * len(tokens)}
        variant, _ = perturb(row, condition, seed=271828)
        noisy_text = " ".join(variant["tokens"])
        noisy_tokens, noisy_offsets = tokenize_with_offsets(noisy_text)
        noisy_tags = model.predict_checked(noisy_tokens)
        noisy = entities(noisy_tokens, noisy_offsets, noisy_tags, noisy_text)
        payload = {"model_id": model_id, "clean": {"text": sentence, "entities": clean},
                   condition: {"text": noisy_text, "entities": noisy}}
        st.subheader("Clean")
        st.markdown(highlight(sentence, clean), unsafe_allow_html=True)
        st.subheader(condition)
        st.markdown(highlight(noisy_text, noisy), unsafe_allow_html=True)
        st.json(payload)
        st.download_button("Download JSON", json.dumps(payload, indent=2),
                           file_name="ner_result.json", mime="application/json")
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, ["condition", "text", "type", "start", "end"])
        writer.writeheader()
        for key, values in (("clean", clean), (condition, noisy)):
            writer.writerows([{"condition": key, **entity} for entity in values])
        st.download_button("Download CSV", buffer.getvalue(),
                           file_name="ner_result.csv", mime="text/csv")
    except (NotImplementedError, ValueError, OSError) as exc:
        st.error(str(exc))

