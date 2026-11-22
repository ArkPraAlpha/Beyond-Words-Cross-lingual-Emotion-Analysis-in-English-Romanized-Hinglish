"""Check that the app's inference path reproduces the saved test predictions.
Usage:  python app/smoke_test.py <checkpoint_dir> <saved_test_logits.csv> <goemotions_ekman_dedup.csv> [n]
The saved logits file is one of outputs/english_emotion/logits/test_logits_*.csv (GoEmotions test)."""
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bw_utils import LABEL_ORDER, MAX_LEN, softmax_np

ckpt, logits_csv, data_csv = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 100
saved = pd.read_csv(logits_csv, dtype={"example_id": str}).head(n)
data = pd.read_csv(data_csv, dtype=str)
data = data[data["split"] == "test"].set_index("example_id")
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()
agree, diffs = 0, []
for _, r in saved.iterrows():
    enc = tok(data.loc[r["example_id"], "text"], truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.inference_mode():
        p = softmax_np(model(**enc).logits[0].float().numpy())
    agree += LABEL_ORDER[int(p.argmax())] == r["pred_label"]
    diffs.append(np.abs(p - r[[f"prob_{l}" for l in LABEL_ORDER]].astype(float).values).max())
print(f"prediction agreement: {agree}/{len(saved)}   mean max |prob diff|: {np.mean(diffs):.4f}")
print("PASS" if agree / len(saved) >= 0.95 else "CHECK: the app does not reproduce the saved predictions")
