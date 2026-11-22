"""Lightweight helpers for the Beyond Words local frontend (no torch/transformers)."""
import json
import math
from pathlib import Path

import numpy as np

LABEL_ORDER = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
MAX_LEN = 128


def load_cards(path: Path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def softmax_np(logits, temperature: float = 1.0):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy_bits(probs) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


def confidence_summary(probs, labels=LABEL_ORDER, low_conf=0.50, small_margin=0.15):
    """Return a heuristic confidence summary; not a calibrated reliability estimate."""
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    margin = float(probs[top] - probs[second])
    flag = "uncertain" if probs[top] < low_conf or margin < small_margin else "confident"
    return {
        "label": labels[top],
        "prob": float(probs[top]),
        "runner_up": labels[second],
        "runner_up_prob": float(probs[second]),
        "margin": margin,
        "entropy_bits": entropy_bits(probs),
        "max_entropy_bits": math.log2(len(labels)),
        "flag": flag,
    }
