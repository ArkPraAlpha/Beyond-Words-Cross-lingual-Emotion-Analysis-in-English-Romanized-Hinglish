"""FastAPI inference service for the Beyond Words Colab runtime.

The service loads one selected checkpoint at a time to reduce Colab GPU memory use.
Set BW_API_KEY and BW_REGISTRY_PATH before starting uvicorn.
"""
from __future__ import annotations

import gc
import json
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

LABEL_ORDER = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
MAX_LEN = 128
REGISTRY_PATH = Path(os.environ.get("BW_REGISTRY_PATH", "/content/bw_checkpoint_registry.json"))
API_KEY = os.environ.get("BW_API_KEY", "")
CONDITION_NAMES = {
    "A": "XLM-R, English only",
    "B": "XLM-R + EmoSen",
    "C": "HingRoBERTa, English only",
    "D": "HingRoBERTa + EmoSen",
}

app = FastAPI(title="Beyond Words inference API", version="1.0.0")
_model_lock = threading.Lock()
_loaded_key: tuple[str, int] | None = None
_tokenizer = None
_model = None
_device = None


def read_registry() -> dict[str, dict[str, str]]:
    if not REGISTRY_PATH.exists():
        raise RuntimeError(f"Checkpoint registry not found: {REGISTRY_PATH}")
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return data.get("conditions", {})


def check_auth(authorization: str | None) -> None:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Server API key is not configured.")
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Missing or incorrect bearer API key.")


class PredictRequest(BaseModel):
    condition: str = Field(pattern="^[ABCD]$")
    seed: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=5000)
    temperature: float = Field(default=1.0, ge=0.5, le=3.0)


def get_loaded_model(condition: str, seed: int):
    global _loaded_key, _tokenizer, _model, _device
    key = (condition, seed)
    if _loaded_key == key and _model is not None:
        return _tokenizer, _model, _device

    registry = read_registry()
    condition_models = registry.get(condition)
    if not condition_models or str(seed) not in condition_models:
        raise HTTPException(status_code=404, detail=f"Checkpoint not available for condition {condition}, seed {seed}.")

    # Only one checkpoint remains resident. Switching conditions/seeds frees the previous model.
    if _model is not None:
        del _model
        del _tokenizer
        _model = None
        _tokenizer = None
        _loaded_key = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    path = Path(condition_models[str(seed)])
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Checkpoint path is no longer available in Colab: {path}")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
        if int(_model.config.num_labels) != len(LABEL_ORDER):
            raise ValueError(
                f"Expected a seven-output checkpoint, found num_labels={_model.config.num_labels} at {path}"
            )
        _model.to(_device)
        _model.eval()
        _loaded_key = key
        return _tokenizer, _model, _device
    except HTTPException:
        raise
    except Exception as exc:
        _model = None
        _tokenizer = None
        _loaded_key = None
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Could not load the selected checkpoint: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "unknown"
    return {"status": "ok", "service": "beyond-words", "device": device}


@app.get("/models")
def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    check_auth(authorization)
    registry = read_registry()
    ordered = []
    for condition in ["D", "C", "B", "A"]:
        if condition in registry:
            ordered.append({
                "condition": condition,
                "name": CONDITION_NAMES[condition],
                "seeds": sorted([int(seed) for seed in registry[condition].keys()]),
            })
    return {"models": ordered, "labels": LABEL_ORDER, "max_length": MAX_LEN}


@app.post("/predict")
def predict(request: PredictRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    check_auth(authorization)
    clean_text = request.text.strip()
    if not clean_text:
        raise HTTPException(status_code=422, detail="Text must contain at least one non-whitespace character.")

    with _model_lock:
        tokenizer, model, device = get_loaded_model(request.condition, request.seed)
        try:
            import torch
            untruncated = tokenizer(clean_text, add_special_tokens=True, truncation=False)["input_ids"]
            truncated = len(untruncated) > MAX_LEN
            encoded = tokenizer(
                clean_text,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            token_count = int(encoded["input_ids"].shape[1])
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits[0].float()
                probabilities = torch.softmax(logits / float(request.temperature), dim=-1).cpu().tolist()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
    return {
        "condition": request.condition,
        "seed": request.seed,
        "labels": LABEL_ORDER,
        "probabilities": [float(x) for x in probabilities],
        "predicted_label": LABEL_ORDER[top_indices[0]],
        "top_probability": float(probabilities[top_indices[0]]),
        "runner_up_label": LABEL_ORDER[top_indices[1]],
        "runner_up_probability": float(probabilities[top_indices[1]]),
        "n_tokens": token_count,
        "truncated": truncated,
        "max_length": MAX_LEN,
        "device": str(device),
        "temperature": float(request.temperature),
    }
