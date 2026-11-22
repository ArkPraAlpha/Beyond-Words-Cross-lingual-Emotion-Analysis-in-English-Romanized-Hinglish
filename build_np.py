"""
Build 03_english_emotion_baseline.ipynb
Run:  python build_nb.py
Out:  E:/Beyond WORDS/notebooks/03_english_emotion_baseline.ipynb
"""
import json, uuid, pathlib

OUT = pathlib.Path("E:/Beyond WORDS/notebooks/03_english_emotion_baseline.ipynb")

def _id():
    return uuid.uuid4().hex[:8]

def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": src}

def code(src: str) -> dict:
    return {"cell_type": "code", "id": _id(), "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

# ─────────────────────────────────────────────────────────────────────────────
C = []   # cells list
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# MARKDOWN: Project Overview
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
# Beyond Words — Notebook 03: English Emotion Classification Baseline

**Research project:** Beyond Words — Cross-lingual Emotion and Sarcasm Analysis in Romanized Hinglish  
**Notebook role:** Experimental Condition A — XLM-R backbone, English-only GoEmotions training  
**Version:** 1.0.0

---

## Research context

This notebook implements **Condition A** in a 2 × 2 research design:

| Condition | Backbone | Task-training data |
|---|---|---|
| **A (this notebook)** | **XLM-R** | **English only** |
| B | XLM-R | English + Hinglish |
| C | HingRoBERTa | English only |
| D | HingRoBERTa | English + Hinglish |

The goal is **not** simply to achieve the highest possible F1 score. The goal is to build a \
**trustworthy, reproducible English baseline** that can be compared fairly against HingRoBERTa \
and Hinglish-augmented experimental conditions.

## What this notebook does

1. Loads the frozen, pre-processed GoEmotions dataset (`goemotions_ekman_dedup.csv`)
2. Validates data integrity and label consistency against documented guarantees
3. Tokenises text using XLM-RoBERTa's SentencePiece tokeniser
4. Trains a seven-class emotion classifier with class-weighted cross-entropy loss
5. Saves the best checkpoint using validation macro-F1
6. Evaluates on the held-out test set exactly once
7. Saves raw logits, softmax probabilities, ECE, and Brier scores
8. Runs three independent seeds and aggregates results

## Seven-class emotion ontology (frozen throughout all conditions)

```
0 → anger    1 → disgust    2 → fear    3 → joy    4 → sadness    5 → surprise    6 → neutral
```

This label mapping **must not change** across any experimental condition.

## Key design decisions

| Decision | Rationale |
|---|---|
| **Macro-F1 as primary metric** | Gives minority classes (fear, disgust) equal weight; accuracy misleads on imbalanced data |
| **Class-weighted loss** | Compensates for Joy/Neutral majority in GoEmotions |
| **Raw logits saved** | Enables calibration, uncertainty analysis, and cross-condition comparison without retraining |
| **Three seeds** | Quantifies run-to-run variability; not a guarantee of statistical precision |
| **Backbone-agnostic design** | Changing `MODEL_NAME` to HingRoBERTa adapts the entire pipeline |

> ⚠️ **Research integrity:** The test set is used **exactly once**, after model selection is complete. \
Never inspect test predictions to choose hyperparameters.
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Installation
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 1 — Installation and dependency versions

We install required packages via `uv` (a fast pip-compatible installer pre-available in Colab).  
All minimum version pins have been verified with Colab's T4 runtime.

If running locally, replace `subprocess.run` calls with `pip install` in a terminal.
"""))

C.append(code("""\
import subprocess, sys

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    subprocess.run(["pip", "install", "-q", "uv"], check=True)
    subprocess.run(
        [
            "uv", "pip", "install", "-q", "-U",
            "transformers>=4.40.0",
            "datasets>=2.19.0",
            "accelerate>=0.29.0",
            "scikit-learn>=1.4.0",
            "pandas>=2.0.0",
            "numpy>=1.26.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "scipy>=1.12.0",
            "tqdm>=4.66.0",
        ],
        check=True,
    )
    print("Packages installed successfully.")
else:
    print("Not in Colab — please ensure requirements are installed in your environment.")
    print("Required: transformers, datasets, accelerate, scikit-learn, pandas,")
    print("          numpy, matplotlib, seaborn, scipy, tqdm")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Imports
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 2 — Imports

All imports are gathered in one cell so the notebook fails fast if a package is missing.  
`TOKENIZERS_PARALLELISM=false` suppresses a benign Hugging Face warning about forked processes.
"""))

C.append(code("""\
# Standard library
import datetime
import json
import os
import platform
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Scientific stack
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax as scipy_softmax

# scikit-learn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Hugging Face Transformers
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"Transformers : {transformers.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")
print(f"Platform     : {platform.platform()}")
print(f"In Colab     : {IN_COLAB}")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Project paths
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 3 — Project paths and directory setup

The project root is `/content/Beyond-Words/` on Google Colab.  
All outputs are written to well-defined subdirectories so raw data is never overwritten.

**Directory layout:**

```
Beyond-Words/
├── data/
│   └── processed/
│       └── goemotions_ekman_dedup.csv   ← frozen; never overwritten
├── models/
│   └── english_emotion/
│       ├── seed_42/
│       ├── seed_123/
│       └── seed_2026/
├── outputs/
│   └── english_emotion/
│       ├── predictions/
│       ├── metrics/
│       ├── logits/
│       ├── figures/
│       └── logs/
└── configs/
    └── english_emotion_config.json
```
"""))

C.append(code("""\
# ── Project root ─────────────────────────────────────────────────────────
try:
    import google.colab  # noqa: F401
    PROJECT_ROOT = Path("/content/Beyond-Words")
except ImportError:
    # Local: assume notebook is in notebooks/, project root is one level up
    PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent

# ── Directory map ─────────────────────────────────────────────────────────
DIRS = {
    "data_processed" : PROJECT_ROOT / "data" / "processed",
    "data_raw"       : PROJECT_ROOT / "data" / "raw",
    "data_interim"   : PROJECT_ROOT / "data" / "interim",
    "models_root"    : PROJECT_ROOT / "models" / "english_emotion",
    "out_predictions": PROJECT_ROOT / "outputs" / "english_emotion" / "predictions",
    "out_metrics"    : PROJECT_ROOT / "outputs" / "english_emotion" / "metrics",
    "out_logits"     : PROJECT_ROOT / "outputs" / "english_emotion" / "logits",
    "out_figures"    : PROJECT_ROOT / "outputs" / "english_emotion" / "figures",
    "out_logs"       : PROJECT_ROOT / "outputs" / "english_emotion" / "logs",
    "configs"        : PROJECT_ROOT / "configs",
}

for name, path in DIRS.items():
    path.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DIRS["data_processed"] / "goemotions_ekman_dedup.csv"

print(f"Project root   : {PROJECT_ROOT}")
print(f"Dataset path   : {DATASET_PATH}")
print(f"Dataset exists : {DATASET_PATH.exists()}")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\\n\\n❌  Dataset not found at: {DATASET_PATH}\\n"
        "Please upload goemotions_ekman_dedup.csv to data/processed/ "
        "before running this notebook.\\n"
        "Do NOT replace it with a different CSV — this dataset is frozen by the "
        "preprocessing pipeline in notebook 01_dataset_exploration.ipynb."
    )

print("\\n✅  All directories created. Dataset located.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Configuration
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 4 — Experiment configuration

All hyperparameters, model choices, and flags live in a single `CFG` dictionary.

### Why centralise configuration?

Changing `MODEL_NAME` from `"FacebookAI/xlm-roberta-base"` to `"l3cube-pune/hing-roberta"` \
(HingRoBERTa) will make the **entire notebook** adapt — tokeniser, model, checkpoint naming — \
without touching any implementation cell. This is the backbone-agnostic design.

### Why these initial hyperparameter values?

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 2e-5 | Standard for fine-tuning pre-trained transformers; too high risks destroying pre-trained representations |
| Epochs | 3 | GoEmotions is large (~39k train); 3 epochs is sufficient and avoids overfitting |
| Batch size | 16 | Fits T4 VRAM (~15 GB) with XLM-R-base at max_seq_len 128; reduce to 8 if OOM |
| max_seq_len | 128 | We will verify empirically that this covers ≥95% of GoEmotions texts |
| Warmup ratio | 0.1 | Linearly ramp LR for the first 10% of steps — prevents corrupting pre-trained weights early |
| Weight decay | 0.01 | L2 regularisation on non-bias/non-LayerNorm parameters |
| Gradient clipping | 1.0 | Prevents exploding gradients, especially in early fine-tuning |
| Seeds | [42, 123, 2026] | Three independent runs to estimate variability |
"""))

C.append(code("""\
# ── Frozen label mapping (must not change across any experimental condition) ──
LABEL_ORDER = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
ID2LABEL    = {i: lbl for i, lbl in enumerate(LABEL_ORDER)}
LABEL2ID    = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
NUM_LABELS  = len(LABEL_ORDER)

# ── Central experiment configuration ─────────────────────────────────────
CFG = {
    # --- Model (backbone-agnostic: change this one string to switch backbone) ---
    "model_name"          : "FacebookAI/xlm-roberta-base",
    "num_labels"          : NUM_LABELS,

    # --- Data ---
    "max_seq_len"         : 128,
    "dataset_path"        : str(DATASET_PATH),

    # --- Training ---
    "num_epochs"          : 3,
    "train_batch_size"    : 16,
    "eval_batch_size"     : 32,
    "learning_rate"       : 2e-5,
    "weight_decay"        : 0.01,
    "warmup_ratio"        : 0.1,
    "grad_clip"           : 1.0,
    "grad_accum_steps"    : 1,     # Increase if OOM: effective_batch = train_batch * grad_accum
    "use_fp16"            : True,  # Mixed precision; auto-disabled if no CUDA
    "use_class_weights"   : True,  # Set False for unweighted ablation

    # --- Experiment control ---
    "seeds"               : [42, 123, 2026],
    "primary_metric"      : "macro_f1",

    # --- Naming ---
    "experiment_name"     : "xlmr_english_ekman7",
    "condition_label"     : "A",
}

# Save config to disk for reproducibility
config_path = DIRS["configs"] / "english_emotion_config.json"
with open(config_path, "w") as f:
    json.dump(
        {**CFG, "label_order": LABEL_ORDER, "id2label": ID2LABEL, "label2id": LABEL2ID},
        f, indent=2,
    )

print("Experiment configuration:")
for k, v in CFG.items():
    print(f"  {k:<26} = {v}")
print(f"\\nConfig saved: {config_path}")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GPU Inspection
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 5 — GPU and environment inspection

### Why check the GPU first?

Google Colab's free T4 GPU has ~15 GB VRAM. Training XLM-R-base at batch size 16 uses \
approximately 7–9 GB. If CUDA is unavailable, training falls back to CPU, which is \
~50× slower — one epoch may take several hours instead of ~12 minutes.

### Mixed precision (FP16)

Instead of storing all weights and activations as 32-bit floats, FP16 uses 16-bit floats \
for most operations. This approximately:
- Halves VRAM consumption
- Speeds up computation via NVIDIA Tensor Cores

PyTorch's `torch.cuda.amp.autocast()` context manager handles FP16 automatically and falls \
back to FP32 for numerically sensitive operations (e.g., softmax).

### Recovery if CUDA out-of-memory (OOM)

If you encounter a CUDA OOM error:
1. Reduce `CFG["train_batch_size"]` from 16 → 8
2. Increase `CFG["grad_accum_steps"]` from 1 → 2 (keeps effective batch = 16)
3. Reduce `CFG["max_seq_len"]` from 128 → 96
4. Restart the runtime and re-run from the top

Do not silently change these settings mid-run — document any deviation.
"""))

C.append(code("""\
print("=" * 60)
print("ENVIRONMENT INSPECTION")
print("=" * 60)

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    DEVICE     = torch.device("cuda")
    gpu_name   = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅  CUDA available")
    print(f"   GPU model  : {gpu_name}")
    print(f"   VRAM total : {gpu_mem_gb:.1f} GB")
    print(f"   Mixed prec : {'FP16 enabled' if CFG['use_fp16'] else 'disabled (CFG override)'}")
else:
    DEVICE = torch.device("cpu")
    CFG["use_fp16"] = False   # FP16 requires CUDA
    print("⚠️  CUDA not available — training on CPU.")
    print("   FP16 disabled automatically.")
    print("   WARNING: CPU training is ~50× slower than T4 GPU.")

print(f"\\n   Selected device  : {DEVICE}")
print(f"   PyTorch version  : {torch.__version__}")
print(f"   Transformers     : {transformers.__version__}")
print(f"   Python           : {sys.version.split()[0]}")
print("=" * 60)
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Reproducibility
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 6 — Reproducibility utilities

### Why fix random seeds?

Neural network training involves many sources of randomness:
- **Weight initialisation** — the classification head starts from random values
- **Data-loading order** — the training DataLoader shuffles examples each epoch
- **Dropout** — randomly zeros activations during training
- **CUDA kernel non-determinism** — some GPU operations are non-deterministic by default

Fixing a seed makes **one specific run** reproducible. Different seeds will — and should — \
produce slightly different results. That is precisely why we run three seeds: to quantify \
variability rather than hide it behind a single lucky run.

### Performance note

`torch.backends.cudnn.deterministic = True` disables some highly-optimised but \
non-deterministic CUDA kernels. This can slow training by 10–30%. This trade-off is \
acceptable for research reproducibility.
"""))

C.append(code("""\
def set_seed(seed: int) -> None:
    \"\"\"Set all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value applied to Python, NumPy, PyTorch CPU, and PyTorch CUDA.
    \"\"\"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🌱  Random seed set to {seed}")

# Set initial seed for dataset inspection (before training seeds)
set_seed(CFG["seeds"][0])
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Dataset Loading
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 7 — Dataset loading

The frozen dataset `goemotions_ekman_dedup.csv` was produced by notebook \
`01_dataset_exploration.ipynb`. We load it **read-only**.

### Confirmed CSV schema (inspected before writing this notebook)

| Column | Type | Description |
|---|---|---|
| `split` | str | `train`, `validation`, or `test` |
| `row_index` | int | Original row position in GoEmotions HF dataset |
| `example_id` | str | Unique Reddit comment identifier from GoEmotions |
| `text` | str | **Original text — used for training** |
| `source_labels` | str | Fine-grained GoEmotions label(s) before Ekman mapping |
| `mapped_label` | str | Seven-class Ekman label (our training target) |
| `keep` | bool | `True` for all rows (excluded examples were removed) |
| `mapping_reason` | str | Why this example was retained |
| `text_norm` | str | Lowercased/whitespace-normalised text (used for dedup checks only) |

### Why we use `text`, not `text_norm`

Lowercasing discards capitalisation signals that carry emotional information \
(e.g., "WHY THE FUCK" vs "why the fuck"). The tokeniser handles casing internally.

### Confirmed dataset counts (from preprocessing audit)

| Split | Count |
|---|---|
| Train | 39,257 |
| Validation | 4,930 |
| Test | 4,962 |
| **Total** | **49,149** |
"""))

C.append(code("""\
print(f"Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH, dtype=str)   # dtype=str prevents silent type coercion

# ── Confirm required columns ──────────────────────────────────────────────
REQUIRED_COLS = ["split", "example_id", "text", "mapped_label", "keep", "text_norm"]
missing_cols  = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"❌  Missing required columns: {missing_cols}\\n"
        f"   Found columns: {df.columns.tolist()}"
    )

# Normalise string columns
df["mapped_label"] = df["mapped_label"].str.strip().str.lower()
df["split"]        = df["split"].str.strip().str.lower()

print(f"\\nDataset shape : {df.shape}")
print(f"All columns   : {df.columns.tolist()}")
print(f"\\nSplit counts:")
for split_name in ["train", "validation", "test"]:
    n = (df["split"] == split_name).sum()
    print(f"  {split_name:>12} : {n:>7,}")
print(f"  {'TOTAL':>12} : {len(df):>7,}")

print(f"\\nSample rows:")
df.head(3)
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Integrity Checks
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 8 — Dataset integrity checks

Before any training, we validate the dataset against the documented preprocessing guarantees. \
If any check fails, training is halted — we do not silently fix or skip bad data.

### Checks performed

1. **Valid split names** — every row is `train`, `validation`, or `test`
2. **No cross-split example_id duplicates** — no `example_id` appears in multiple splits
3. **No text-level train/test leakage** — no normalised text exists in both train and test
4. **Valid labels** — every `mapped_label` is in the frozen seven-class ontology
5. **No null values** in critical columns
6. **Expected total count** — the documented 49,149 retained examples
7. **`keep` column is all True** — the CSV should contain only retained examples

### Why are these checks necessary?

The deduplication pipeline preserved train/test separation. We confirm that guarantee \
here rather than trusting it silently — a subtle data-leakage bug could make results \
appear artificially better and invalidate cross-condition comparisons.
"""))

C.append(code("""\
print("=" * 60)
print("DATASET INTEGRITY CHECKS")
print("=" * 60)

errors        = []
warnings_list = []

# ── Check 1: Valid split names ─────────────────────────────────────────────
valid_splits   = {"train", "validation", "test"}
invalid_splits = set(df["split"].unique()) - valid_splits
if invalid_splits:
    errors.append(f"Invalid split names: {invalid_splits}")
else:
    print("✅  Check 1: All split names are valid")

# ── Check 2: No cross-split example_id duplicates ─────────────────────────
id_split_counts = df.groupby("example_id")["split"].nunique()
cross_split_ids = id_split_counts[id_split_counts > 1]
if len(cross_split_ids) > 0:
    errors.append(f"{len(cross_split_ids)} example_ids appear in multiple splits")
else:
    print("✅  Check 2: No example_id appears in multiple splits")

# ── Check 3: No cross-split text-level leakage ────────────────────────────
train_norms = set(df[df["split"] == "train"]["text_norm"].dropna())
val_norms   = set(df[df["split"] == "validation"]["text_norm"].dropna())
test_norms  = set(df[df["split"] == "test"]["text_norm"].dropna())

train_test_overlap = train_norms & test_norms
train_val_overlap  = train_norms & val_norms

if train_test_overlap:
    errors.append(f"Train/test text leakage: {len(train_test_overlap)} overlapping normalised texts")
else:
    print("✅  Check 3: No train/test text leakage detected")

if train_val_overlap:
    warnings_list.append(
        f"{len(train_val_overlap)} normalised texts appear in both train and validation "
        "(review preprocessing audit — this is a warning, not an error)"
    )

# ── Check 4: Valid labels ──────────────────────────────────────────────────
invalid_labels = set(df["mapped_label"].unique()) - set(LABEL_ORDER)
if invalid_labels:
    errors.append(f"Invalid labels found: {invalid_labels}")
else:
    print("✅  Check 4: All mapped_label values are in the seven-class ontology")

# ── Check 5: No null values in critical columns ────────────────────────────
null_counts = df[REQUIRED_COLS].isnull().sum()
critical_nulls = null_counts[null_counts > 0]
if len(critical_nulls) > 0:
    errors.append(f"Null values found:\\n{critical_nulls.to_string()}")
else:
    print("✅  Check 5: No null values in critical columns")

# ── Check 6: Expected total count ─────────────────────────────────────────
EXPECTED_TOTAL = 49149
if len(df) != EXPECTED_TOTAL:
    warnings_list.append(
        f"Row count {len(df):,} ≠ expected {EXPECTED_TOTAL:,}. "
        "Review preprocessing audit before proceeding."
    )
    print(f"⚠️   Check 6: Row count mismatch (got {len(df):,}, expected {EXPECTED_TOTAL:,})")
else:
    print(f"✅  Check 6: Total row count matches expected ({EXPECTED_TOTAL:,})")

# ── Check 7: keep column is all True ──────────────────────────────────────
non_kept = (df["keep"].astype(str).str.strip().str.lower() != "true").sum()
if non_kept > 0:
    errors.append(f"{non_kept} rows have keep ≠ True — excluded examples should not be in this CSV")
else:
    print("✅  Check 7: All rows have keep=True (excluded examples correctly removed)")

# ── Report results ─────────────────────────────────────────────────────────
print()
for w in warnings_list:
    print(f"⚠️   WARNING: {w}")

if errors:
    print()
    for e in errors:
        print(f"❌  ERROR: {e}")
    raise RuntimeError(
        "\\n\\nDataset integrity checks FAILED. "
        "Do not proceed until the issues above are resolved. "
        "These checks protect research integrity."
    )

print("\\n✅  All integrity checks passed. Dataset is safe to use for training.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Label Validation & Distribution
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 9 — Label validation and split distribution analysis

### Why analyse class distributions before training?

GoEmotions is severely imbalanced:
- **Joy** and **Neutral** dominate because Reddit comments more often express positive sentiment \
or no clear emotion
- **Fear** and **Disgust** are rare (< 2% and < 2% of training data respectively)

A naive model that always predicts Joy or Neutral can achieve ~38% accuracy. This is why \
**accuracy is a misleading metric** for this task, and **macro-F1 is essential**.

We also verify that class distributions are roughly comparable across splits — if all Fear \
examples ended up in test but not validation, validation metrics would not reliably predict \
test performance.
"""))

C.append(code("""\
print("=" * 60)
print("LABEL DISTRIBUTION ACROSS SPLITS")
print("=" * 60)

split_label_counts = {}
for split in ["train", "validation", "test"]:
    sdf    = df[df["split"] == split]
    counts = {lbl: int((sdf["mapped_label"] == lbl).sum()) for lbl in LABEL_ORDER}
    split_label_counts[split] = counts

# Print table
header = f"{'Emotion':<12}" + "".join(f"  {s:>12}" for s in ["Train", "Validation", "Test", "Total"])
print(header)
print("-" * len(header))
for lbl in LABEL_ORDER:
    tr = split_label_counts["train"][lbl]
    va = split_label_counts["validation"][lbl]
    te = split_label_counts["test"][lbl]
    tot = tr + va + te
    print(f"{lbl:<12}  {tr:>12,}  {va:>12,}  {te:>12,}  {tot:>12,}")

totals    = [sum(split_label_counts[s][l] for l in LABEL_ORDER) for s in ["train","validation","test"]]
grand_tot = sum(totals)
print("-" * len(header))
print(f"{'TOTAL':<12}  {totals[0]:>12,}  {totals[1]:>12,}  {totals[2]:>12,}  {grand_tot:>12,}")

train_counts  = split_label_counts["train"]
max_class     = max(train_counts, key=train_counts.get)
min_class     = min(train_counts, key=train_counts.get)
imb_ratio     = train_counts[max_class] / train_counts[min_class]
print(f"\\n📊  Majority class (train) : {max_class} ({train_counts[max_class]:,})")
print(f"📊  Minority class (train) : {min_class} ({train_counts[min_class]:,})")
print(f"📊  Imbalance ratio        : {imb_ratio:.1f}× — class weighting is essential")
"""))

C.append(code("""\
# ── Visualise class distributions ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
colors = plt.cm.Set2(np.linspace(0, 1, NUM_LABELS))

for ax, split in zip(axes, ["train", "validation", "test"]):
    counts = [split_label_counts[split][lbl] for lbl in LABEL_ORDER]
    bars   = ax.bar(LABEL_ORDER, counts, color=colors)
    ax.set_title(f"{split.capitalize()} (n={sum(counts):,})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Emotion class")
    ax.set_ylabel("Examples")
    ax.tick_params(axis="x", rotation=30)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{cnt:,}", ha="center", va="bottom", fontsize=8)

plt.suptitle("GoEmotions Ekman-7 — Class Distribution per Split",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = DIRS["out_figures"] / "class_distribution.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figure saved: {fig_path}")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Tokeniser Loading
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 10 — Tokeniser loading

### What is a tokeniser?

A tokeniser converts raw text into a sequence of integer **token IDs** that the model can \
process. XLM-RoBERTa uses **SentencePiece** tokenisation with a vocabulary of 250,002 tokens \
covering 100 languages.

The tokeniser also produces:
- **`input_ids`** — the integer token IDs (including special tokens `<s>` and `</s>`)
- **`attention_mask`** — 1 for real tokens, 0 for padding; the model ignores padded positions

### Why preserve original text (not `text_norm`)?

XLM-RoBERTa's vocabulary includes both cased and uncased tokens. Lowercasing discards \
capitalisation signals that carry emotional weight ("WHY" vs "why"). We use the \
original `text` column, not the normalised `text_norm`.
"""))

C.append(code("""\
print(f"Loading tokeniser: {CFG['model_name']}")
tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

print(f"Tokeniser class  : {type(tokenizer).__name__}")
print(f"Vocabulary size  : {tokenizer.vocab_size:,}")
print(f"Model max length : {tokenizer.model_max_length}")
print(f"Config max_seq_len: {CFG['max_seq_len']}")

# Sanity check on a sample with emotional content
sample_text   = "WHY THE FUCK IS THIS HAPPENING?! 😡"
sample_tokens = tokenizer(sample_text, return_tensors="pt")
token_strings = tokenizer.convert_ids_to_tokens(sample_tokens["input_ids"][0].tolist())
print(f"\\nSample            : {sample_text!r}")
print(f"Token ids shape   : {sample_tokens['input_ids'].shape}")
print(f"Tokens            : {token_strings}")
print("\\n✅  Tokeniser loaded successfully.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Token-Length Analysis
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 11 — Token-length analysis

### Why analyse token lengths?

The `max_seq_len` parameter **truncates** any text longer than 128 tokens. If many examples \
are longer than 128 tokens, truncation silently discards the end of those texts, potentially \
removing emotional content that appears at the end of a comment.

We compute:
- **Mean / median / 95th percentile** token length — to understand the distribution
- **Percentage truncated** — examples where `len(tokens) > max_seq_len`

### Decision rule

If > 10% of examples are truncated, consider increasing `max_seq_len` to 256. However, \
this roughly **doubles VRAM usage and computation cost**, and may exceed T4 capacity \
at batch size 16. The percentage truncated is the key number to report.

> This analysis takes ~1–2 minutes for 49k examples. \
Token lengths are computed without padding or truncation to measure actual lengths.
"""))

C.append(code("""\
print("Computing token lengths on the full dataset (~1-2 minutes)...")

def get_token_lengths(texts: List[str], batch_size: int = 512) -> List[int]:
    \"\"\"Tokenise texts in batches (no padding/truncation) and return raw lengths.\"\"\"
    lengths = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(batch, add_special_tokens=True, truncation=False, padding=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths

all_lengths    = get_token_lengths(df["text"].tolist())
df["token_len"] = all_lengths

for split in ["train", "validation", "test"]:
    lens          = df[df["split"] == split]["token_len"].values
    truncated_pct = (lens > CFG["max_seq_len"]).mean() * 100
    print(f"\\n{split.upper()} (n={len(lens):,}):")
    print(f"  Mean              : {lens.mean():.1f} tokens")
    print(f"  Median            : {np.median(lens):.1f} tokens")
    print(f"  95th percentile   : {np.percentile(lens, 95):.1f} tokens")
    print(f"  Max               : {lens.max()} tokens")
    print(f"  % truncated (>{CFG['max_seq_len']}) : {truncated_pct:.2f}%")
"""))

C.append(code("""\
fig, ax = plt.subplots(figsize=(11, 4))
ax.hist(df["token_len"], bins=80, color="steelblue", alpha=0.7, edgecolor="white")
ax.axvline(CFG["max_seq_len"], color="crimson", linestyle="--", linewidth=2.0,
           label=f"max_seq_len = {CFG['max_seq_len']}")
p95 = np.percentile(df["token_len"], 95)
ax.axvline(p95, color="darkorange", linestyle="--", linewidth=1.5,
           label=f"P95 = {p95:.0f} tokens")
ax.set_xlabel("Token length (including special tokens)")
ax.set_ylabel("Number of examples")
ax.set_title("Token-Length Distribution — GoEmotions Ekman-7 (all splits combined)")
ax.legend()
plt.tight_layout()
fig_path = DIRS["out_figures"] / "token_length_distribution.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figure saved: {fig_path}")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Class Weights
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 12 — Class-weight calculation

### Why ordinary cross-entropy fails on imbalanced data

Standard cross-entropy treats every example equally. Because Joy accounts for ~39% of \
training examples, the model is *rewarded* proportionally more for correctly classifying Joy. \
It may learn to favour Joy and Neutral at the expense of rare classes (Fear: ~1.8%, \
Disgust: ~1.6%).

### Class-weighted cross-entropy

We multiply the loss of each example by a weight inversely proportional to its class frequency:

$$w_c = \\frac{N}{K \\cdot n_c}$$

where:
- $N$ = total training examples
- $K$ = number of classes (7)
- $n_c$ = number of training examples in class $c$

A rare class like Fear gets a **higher weight**, so the optimiser is penalised more heavily \
for misclassifying Fear examples.

### Important caveats

- Class weighting changes the training objective but does **not guarantee** better macro-F1
- We will compare weighted vs unweighted as an ablation (set `CFG["use_class_weights"] = False`)
- Weights are computed **only from the training split** — never from validation or test data
"""))

C.append(code("""\
# ── Compute class weights from training split only ─────────────────────────
# (Never compute from validation or test — that would be data leakage)
train_df         = df[df["split"] == "train"].copy()
train_labels_arr = train_df["mapped_label"].map(LABEL2ID).values  # integer array

# sklearn's balanced scheme: N / (K * count_c)
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_LABELS),
    y=train_labels_arr,
)

CLASS_WEIGHTS_TENSOR = torch.tensor(class_weights_array, dtype=torch.float32).to(DEVICE)

print(f"Class weights computed from {len(train_df):,} training examples\\n")
print(f"{'Emotion':<12}  {'Train n':>10}  {'Weight':>10}")
print("-" * 38)
for i, lbl in enumerate(LABEL_ORDER):
    n = int((train_df["mapped_label"] == lbl).sum())
    print(f"{lbl:<12}  {n:>10,}  {class_weights_array[i]:>10.4f}")

print(f"\\nNote: unweighted baseline uses weight = 1.0 for all classes.")
print(f"      Set CFG['use_class_weights'] = False to run the ablation.")
print(f"\\n✅  Class weights tensor on device: {CLASS_WEIGHTS_TENSOR.device}")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — Dataset & DataLoader
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 13 — Dataset and DataLoader preparation

### PyTorch Dataset vs DataLoader

- A **`Dataset`** knows how to retrieve **one example** (tokenise text → tensors)
- A **`DataLoader`** batches multiple examples, shuffles training data each epoch, \
and uses parallel CPU workers for fast data loading

### Dynamic padding with DataCollatorWithPadding

Instead of padding every example to `max_seq_len` (128) regardless of actual length, \
`DataCollatorWithPadding` pads each **batch** to the longest sequence in that batch. \
This saves computation when most examples in a batch are short (the common case in \
GoEmotions).

### Why shuffle only training data?

Validation and test evaluation must be **deterministic** — identical results every time. \
Shuffling would change batch composition and could cause minor metric differences.
"""))

C.append(code("""\
# ── EmotionDataset ────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    \"\"\"PyTorch Dataset for seven-class emotion classification.

    Stores raw texts and example_ids alongside tensors for error analysis.
    Tokenisation uses truncation but NOT padding — dynamic padding is handled
    per-batch by EmotionDataCollator.
    \"\"\"

    def __init__(
        self,
        texts       : List[str],
        labels      : List[int],
        example_ids : List[str],
        tokenizer   : AutoTokenizer,
        max_len     : int,
    ):
        self.texts       = texts
        self.labels      = labels
        self.example_ids = example_ids
        self.tokenizer   = tokenizer
        self.max_len     = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation  = True,
            max_length  = self.max_len,
            padding     = False,     # padding handled by DataCollator
            return_tensors = "pt",
        )
        return {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "labels"         : torch.tensor(self.labels[idx], dtype=torch.long),
            "example_id"     : self.example_ids[idx],  # str — not a tensor
        }


# ── EmotionDataCollator ───────────────────────────────────────────────────
class EmotionDataCollator:
    \"\"\"Collator that pads tensors per batch and passes string fields through.\"\"\"

    def __init__(self, tokenizer: AutoTokenizer):
        self._pad = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def __call__(self, features: List[Dict]) -> Dict:
        # Pop string fields before passing to DataCollatorWithPadding
        example_ids = [f.pop("example_id") for f in features]
        batch       = self._pad(features)
        batch["example_id"] = example_ids
        return batch


# ── build_datasets ────────────────────────────────────────────────────────
def build_datasets(df: pd.DataFrame, tokenizer: AutoTokenizer, cfg: Dict) -> Dict:
    \"\"\"Build EmotionDataset objects for all three splits.\"\"\"
    datasets = {}
    for split in ["train", "validation", "test"]:
        sdf    = df[df["split"] == split].reset_index(drop=True)
        texts  = sdf["text"].tolist()
        labels = sdf["mapped_label"].map(LABEL2ID).tolist()
        ids    = sdf["example_id"].tolist()
        datasets[split] = EmotionDataset(texts, labels, ids, tokenizer, cfg["max_seq_len"])
    return datasets


# ── make_dataloaders ──────────────────────────────────────────────────────
def make_dataloaders(datasets: Dict, cfg: Dict, collator: EmotionDataCollator) -> Dict:
    \"\"\"Wrap datasets in DataLoaders.\"\"\"
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size  = cfg["train_batch_size"],
            shuffle     = True,
            collate_fn  = collator,
            num_workers = 2,
            pin_memory  = CUDA_AVAILABLE,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size  = cfg["eval_batch_size"],
            shuffle     = False,
            collate_fn  = collator,
            num_workers = 2,
            pin_memory  = CUDA_AVAILABLE,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size  = cfg["eval_batch_size"],
            shuffle     = False,
            collate_fn  = collator,
            num_workers = 2,
            pin_memory  = CUDA_AVAILABLE,
        ),
    }


# Build and report
print("Building datasets...")
DATASETS = build_datasets(df, tokenizer, CFG)
for split_name, ds in DATASETS.items():
    print(f"  {split_name:>12} : {len(ds):>7,} examples")
print("\\n✅  Datasets ready.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — Model Architecture & Training Utilities
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 14 — Model architecture and training utilities

### Architecture overview

```
Input text
   ↓
Tokeniser  (SentencePiece, vocab 250,002)
   ↓
XLM-RoBERTa-base Transformer  (12 layers · 768 hidden dim · 12 attention heads · 125M params)
   ↓
[CLS] token hidden state  (768-dim vector — summarises the whole sequence)
   ↓
Dropout (0.1)
   ↓
Linear layer  (768 → 7)   ← randomly initialised classification head
   ↓
Raw logits  (7 values, unbounded)
   ↓   argmax → predicted class
   ↓   softmax → probabilities (sum to 1)
```

### What are logits?

Logits are the **raw, unnormalised scores** from the final linear layer. They can be any \
real number. A higher logit means higher confidence for that class.

**Softmax** converts logits to probabilities:

$$p_c = \\frac{e^{z_c}}{\\sum_{j=1}^{7} e^{z_j}}$$

**Why save raw logits (not just predicted labels)?**

Two models can predict the same class with very different confidence:
- Model A: Joy logit = 4.5, Joy probability = 0.96
- Model B: Joy logit = 0.8, Joy probability = 0.42

If only the label "Joy" is saved, the confidence difference is permanently lost. Saving \
logits enables calibration analysis, uncertainty quantification on Hinglish data, and \
ensemble methods.

### Training step

Each training step:
1. **Forward pass**: compute logits and loss
2. **`loss.backward()`**: compute gradients via backpropagation
3. **Gradient clipping**: cap gradient norm at 1.0 to prevent instability
4. **`optimiser.step()`**: update weights using AdamW
5. **`scheduler.step()`**: adjust learning rate according to warmup/decay schedule

### Why AdamW (not Adam)?

AdamW decouples weight decay from the gradient update, which is more principled and \
produces better-calibrated regularisation for Transformer fine-tuning.

### Learning-rate schedule

Linear warmup (0 → 2e-5 over 10% of steps) followed by linear decay (2e-5 → 0 over \
remaining 90%). Warmup prevents corrupting pre-trained representations with high initial \
learning rates when the classification head weights are still random.
"""))

C.append(code("""\
# ── Metric computation ────────────────────────────────────────────────────
def compute_metrics(
    true_labels : List[int],
    pred_labels : List[int],
    prefix      : str = "",
) -> Dict[str, float]:
    \"\"\"Compute accuracy, macro-F1, and weighted-F1.

    Parameters
    ----------
    true_labels : ground-truth integer class indices
    pred_labels : predicted integer class indices
    prefix      : key prefix, e.g. 'val_' or 'test_'

    Returns
    -------
    dict mapping metric name → float value
    \"\"\"
    return {
        f"{prefix}accuracy"    : accuracy_score(true_labels, pred_labels),
        f"{prefix}macro_f1"    : f1_score(true_labels, pred_labels, average="macro",    zero_division=0),
        f"{prefix}weighted_f1" : f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
    }


# ── Training epoch ────────────────────────────────────────────────────────
def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    optimiser : torch.optim.Optimizer,
    scheduler,
    loss_fn   : nn.Module,
    cfg       : Dict,
    scaler    : Optional[torch.cuda.amp.GradScaler],
    epoch     : int,
) -> float:
    \"\"\"One full training epoch. Returns mean training loss.\"\"\"
    model.train()
    total_loss = 0.0
    n_steps    = 0
    optimiser.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)

        if cfg["use_fp16"] and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
                loss   = loss_fn(logits, labels) / cfg["grad_accum_steps"]
            scaler.scale(loss).backward()
            if (step + 1) % cfg["grad_accum_steps"] == 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                optimiser.zero_grad()
        else:
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
            loss   = loss_fn(logits, labels) / cfg["grad_accum_steps"]
            loss.backward()
            if (step + 1) % cfg["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()

        total_loss += loss.item() * cfg["grad_accum_steps"]
        n_steps    += 1

        if step % 200 == 0:
            print(f"    Epoch {epoch} | Step {step:>5}/{len(loader)} "
                  f"| Loss {loss.item() * cfg['grad_accum_steps']:.4f}")

    return total_loss / n_steps


# ── Evaluation epoch ──────────────────────────────────────────────────────
def evaluate(
    model   : nn.Module,
    loader  : DataLoader,
    loss_fn : nn.Module,
    cfg     : Dict,
) -> Tuple[float, List[int], List[int], np.ndarray, List[str]]:
    \"\"\"Evaluate model on a DataLoader.

    Returns
    -------
    mean_loss   : float
    true_labels : List[int]
    pred_labels : List[int]
    all_logits  : np.ndarray of shape (N, 7)  — raw logits for all examples
    example_ids : List[str]
    \"\"\"
    model.eval()
    total_loss  = 0.0
    n_steps     = 0
    true_labels = []
    pred_labels = []
    all_logits  = []
    example_ids = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["labels"].to(DEVICE)
            ex_ids    = batch["example_id"]

            if cfg["use_fp16"]:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
                    loss   = loss_fn(logits, labels)
            else:
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
                loss   = loss_fn(logits, labels)

            total_loss  += loss.item()
            n_steps     += 1
            true_labels .extend(labels.cpu().numpy().tolist())
            pred_labels .extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_logits  .append(logits.float().cpu().numpy())
            example_ids .extend(ex_ids)

    return total_loss / n_steps, true_labels, pred_labels, np.vstack(all_logits), example_ids


print("✅  Training and evaluation functions defined.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 15 — Main Training Loop
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 15 — Main training function

### Checkpoint selection strategy

After each epoch we evaluate on the **validation set** and record **validation macro-F1**. \
If macro-F1 improves, we:
1. Save model weights + tokeniser + config to `models/english_emotion/seed_{seed}/`
2. Record the best epoch and macro-F1 value

At the end of training we **reload the best checkpoint** and evaluate on the test set.

### Why macro-F1 for checkpoint selection?

Macro-F1 computes F1 for each of the 7 classes and averages them equally:

$$F1_{\\text{macro}} = \\frac{1}{7} \\sum_{c=1}^{7} F1_c$$

A model that achieves 90% F1 on Joy/Neutral but 10% on Fear would score \
~70% macro-F1 — clearly signalling poor minority-class performance. \
Accuracy would hide this problem.

### Expected training time (T4 GPU, batch 16, FP16, max_seq_len 128)

| Epochs | Approximate time |
|---|---|
| 1 | ~12–15 minutes |
| 3 | ~35–45 minutes |
"""))

C.append(code("""\
def run_training(
    seed          : int,
    cfg           : Dict,
    df            : pd.DataFrame,
    tokenizer     : AutoTokenizer,
    class_weights : Optional[torch.Tensor],
) -> Dict:
    \"\"\"Complete training run for one random seed.

    Parameters
    ----------
    seed          : integer seed for this independent run
    cfg           : experiment configuration dictionary
    df            : frozen dataset DataFrame
    tokenizer     : pre-loaded tokeniser (shared across runs)
    class_weights : weight tensor on DEVICE, or None for unweighted

    Returns
    -------
    dict with keys: seed, best_epoch, best_val_macro_f1, test_metrics,
                    history, checkpoint_dir, test_pred_df, val_pred_df
    \"\"\"
    set_seed(seed)
    print(f"\\n{'='*60}")
    print(f"TRAINING RUN — Seed {seed}  |  Condition {cfg['condition_label']}")
    print(f"{'='*60}")

    # ── Directories ───────────────────────────────────────────────────────
    seed_model_dir = DIRS["models_root"] / f"seed_{seed}"
    seed_model_dir.mkdir(parents=True, exist_ok=True)

    # ── DataLoaders (recreated per seed to get fresh shuffle state) ───────
    local_datasets = build_datasets(df, tokenizer, cfg)
    local_collator = EmotionDataCollator(tokenizer)
    loaders        = make_dataloaders(local_datasets, cfg, local_collator)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\\nLoading model: {cfg['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels = cfg["num_labels"],
        id2label   = ID2LABEL,
        label2id   = LABEL2ID,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    # ── Loss function ──────────────────────────────────────────────────────
    if cfg["use_class_weights"] and class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print("Loss: class-weighted CrossEntropyLoss")
    else:
        loss_fn = nn.CrossEntropyLoss()
        print("Loss: unweighted CrossEntropyLoss (ablation mode)")

    # ── Optimiser — AdamW with weight decay on non-bias/non-LayerNorm params
    no_decay    = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimiser = torch.optim.AdamW(param_groups, lr=cfg["learning_rate"])

    # ── LR scheduler — linear warmup then linear decay ────────────────────
    total_steps  = len(loaders["train"]) * cfg["num_epochs"] // cfg["grad_accum_steps"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps  = warmup_steps,
        num_training_steps = total_steps,
    )
    print(f"Total steps: {total_steps}  |  Warmup steps: {warmup_steps}")

    # ── Mixed-precision scaler ────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if cfg["use_fp16"] else None

    # ── Training loop ──────────────────────────────────────────────────────
    history           = []
    best_macro_f1     = -1.0
    best_epoch        = -1
    best_val_pred_df  = None

    for epoch in range(1, cfg["num_epochs"] + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optimiser,
                                     scheduler, loss_fn, cfg, scaler, epoch)
        val_loss, val_true, val_pred, val_logits, val_ids = evaluate(
            model, loaders["validation"], loss_fn, cfg)
        val_metrics = compute_metrics(val_true, val_pred, prefix="val_")
        elapsed     = time.time() - t0

        print(f"\\n  Epoch {epoch}/{cfg['num_epochs']} summary:")
        print(f"    Train loss   : {train_loss:.4f}")
        print(f"    Val loss     : {val_loss:.4f}")
        print(f"    Val accuracy : {val_metrics['val_accuracy']:.4f}")
        print(f"    Val macro-F1 : {val_metrics['val_macro_f1']:.4f}  ← checkpoint metric")
        print(f"    Val wtd-F1   : {val_metrics['val_weighted_f1']:.4f}")
        print(f"    Epoch time   : {elapsed:.0f}s")

        history.append({
            "epoch"       : epoch,
            "train_loss"  : train_loss,
            "val_loss"    : val_loss,
            **val_metrics,
            "epoch_time_s": elapsed,
        })

        # ── Save best checkpoint ──────────────────────────────────────────
        if val_metrics["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["val_macro_f1"]
            best_epoch    = epoch
            model.save_pretrained(seed_model_dir)
            tokenizer.save_pretrained(seed_model_dir)

            # Build validation predictions dataframe
            val_probs    = scipy_softmax(val_logits, axis=1)
            best_val_pred_df = pd.DataFrame({
                "example_id" : val_ids,
                "true_label" : [ID2LABEL[i] for i in val_true],
                "pred_label" : [ID2LABEL[i] for i in val_pred],
                "correct"    : [t == p for t, p in zip(val_true, val_pred)],
                **{f"logit_{ID2LABEL[i]}": val_logits[:, i] for i in range(NUM_LABELS)},
                **{f"prob_{ID2LABEL[i]}":  val_probs[:, i]  for i in range(NUM_LABELS)},
            })
            # Overwrite val logits file with best-epoch version
            val_logit_path = DIRS["out_logits"] / f"val_logits_seed{seed}_best_epoch{epoch}.csv"
            best_val_pred_df.to_csv(val_logit_path, index=False)

            print(f"  ✅  New best checkpoint saved "
                  f"(val macro-F1 = {best_macro_f1:.4f}, epoch {epoch})")

    # ── Reload best checkpoint → evaluate on test set ─────────────────────
    print(f"\\nReloading best checkpoint (epoch {best_epoch}, "
          f"val macro-F1 = {best_macro_f1:.4f})")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(seed_model_dir)).to(DEVICE)

    test_loss, test_true, test_pred, test_logits, test_ids = evaluate(
        best_model, loaders["test"], loss_fn, cfg)
    test_metrics = compute_metrics(test_true, test_pred, prefix="test_")

    print(f"\\n{'='*60}")
    print(f"TEST SET RESULTS — Seed {seed}")
    print(f"{'='*60}")
    print(f"  Test loss        : {test_loss:.4f}")
    print(f"  Test accuracy    : {test_metrics['test_accuracy']:.4f}")
    print(f"  Test macro-F1    : {test_metrics['test_macro_f1']:.4f}")
    print(f"  Test weighted-F1 : {test_metrics['test_weighted_f1']:.4f}")
    print(classification_report(
        test_true, test_pred,
        target_names = LABEL_ORDER,
        labels       = list(range(NUM_LABELS)),
        zero_division = 0,
    ))

    # ── Save test logits ───────────────────────────────────────────────────
    test_probs   = scipy_softmax(test_logits, axis=1)
    test_pred_df = pd.DataFrame({
        "example_id" : test_ids,
        "true_label" : [ID2LABEL[i] for i in test_true],
        "pred_label" : [ID2LABEL[i] for i in test_pred],
        "correct"    : [t == p for t, p in zip(test_true, test_pred)],
        **{f"logit_{ID2LABEL[i]}": test_logits[:, i] for i in range(NUM_LABELS)},
        **{f"prob_{ID2LABEL[i]}":  test_probs[:, i]  for i in range(NUM_LABELS)},
    })
    test_logit_path = DIRS["out_logits"] / f"test_logits_seed{seed}.csv"
    test_pred_df.to_csv(test_logit_path, index=False)
    print(f"Test logits saved: {test_logit_path}")

    # ── Save training history ──────────────────────────────────────────────
    history_path = DIRS["out_logs"] / f"training_history_seed{seed}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Save run metadata ──────────────────────────────────────────────────
    run_meta = {
        "seed"             : seed,
        "best_epoch"       : best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "test_metrics"     : test_metrics,
        "checkpoint_dir"   : str(seed_model_dir),
        "label_order"      : LABEL_ORDER,
        "cfg"              : {k: str(v) for k, v in cfg.items()},
    }
    with open(seed_model_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return {
        "seed"             : seed,
        "best_epoch"       : best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "test_metrics"     : test_metrics,
        "history"          : history,
        "checkpoint_dir"   : str(seed_model_dir),
        "test_pred_df"     : test_pred_df,
        "val_pred_df"      : best_val_pred_df,
    }


print("✅  run_training() function defined. Ready to execute.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 16 — Calibration
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 16 — Calibration analysis

### What is calibration?

A well-calibrated model should assign probability 0.80 to predictions it is correct on \
exactly 80% of the time. If a model predicts 0.95 confidence but is only correct 60% of \
the time, it is **overconfident**.

This matters for Beyond Words because overconfident predictions on Hinglish examples would \
mask genuine uncertainty — confidence is a signal for disagreement between surface sentiment \
and intended emotion.

### Expected Calibration Error (ECE)

We divide predictions into $B$ equal-width probability bins. For each bin $b$:

$$\\text{ECE} = \\sum_{b=1}^{B} \\frac{|\\mathcal{B}_b|}{N} \\left| \\text{acc}(\\mathcal{B}_b) - \\text{conf}(\\mathcal{B}_b) \\right|$$

A perfect calibration gives ECE = 0. Typical fine-tuned transformers have ECE ≈ 0.05–0.15. \
We use 15 equal-width bins.

### Brier Score (multiclass)

$$\\text{BS} = \\frac{1}{N} \\sum_{i=1}^{N} \\sum_{c=1}^{K} (p_{ic} - y_{ic})^2$$

where $y_{ic} = 1$ if example $i$ belongs to class $c$, else 0. Lower is better.

> ⚠️ Calibration analysis uses **validation probabilities only** for any fitting. \
Raw uncalibrated predictions are always preserved.
"""))

C.append(code("""\
def compute_ece(
    true_labels : List[int],
    probs       : np.ndarray,
    n_bins      : int = 15,
) -> float:
    \"\"\"Expected Calibration Error.

    Uses max-probability (confidence) as the binning dimension.

    Parameters
    ----------
    true_labels : ground-truth integer class indices (N,)
    probs       : softmax probability matrix (N, K)
    n_bins      : number of equal-width confidence bins (default 15)

    Returns
    -------
    float : ECE score (lower = better; 0 = perfect calibration)
    \"\"\"
    max_probs   = probs.max(axis=1)
    pred_labels = probs.argmax(axis=1)
    correct     = (pred_labels == np.array(true_labels)).astype(float)
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    ece         = 0.0
    n_total     = len(true_labels)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (max_probs >= lo) & (max_probs < hi)
        if in_bin.sum() == 0:
            continue
        acc_b = correct[in_bin].mean()
        con_b = max_probs[in_bin].mean()
        ece  += (in_bin.sum() / n_total) * abs(acc_b - con_b)

    return float(ece)


def compute_brier_score(
    true_labels : List[int],
    probs       : np.ndarray,
    num_classes : int,
) -> float:
    \"\"\"Multiclass Brier Score.

    Parameters
    ----------
    true_labels : ground-truth integer class indices (N,)
    probs       : softmax probability matrix (N, K)
    num_classes : number of classes K

    Returns
    -------
    float : mean Brier score (lower = better; 0 = perfect)
    \"\"\"
    n       = len(true_labels)
    one_hot = np.zeros((n, num_classes), dtype=float)
    for i, lbl in enumerate(true_labels):
        one_hot[i, lbl] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def calibration_plot(
    true_labels : List[int],
    probs       : np.ndarray,
    split_name  : str,
    seed        : int,
    n_bins      : int = 15,
    save_dir    : Optional[Path] = None,
) -> None:
    \"\"\"Reliability diagram + confidence histogram.\"\"\"
    max_probs   = probs.max(axis=1)
    pred_labels = probs.argmax(axis=1)
    correct     = (pred_labels == np.array(true_labels)).astype(float)
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    bin_centers, acc_per_bin, conf_per_bin = [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (max_probs >= lo) & (max_probs < hi)
        if in_bin.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        acc_per_bin.append(correct[in_bin].mean())
        conf_per_bin.append(max_probs[in_bin].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.bar(bin_centers, acc_per_bin, width=1 / n_bins * 0.9,
            alpha=0.6, color="steelblue", label="Accuracy per bin")
    ax1.set_xlabel("Mean confidence (max probability)")
    ax1.set_ylabel("Fraction correct")
    ax1.set_title(f"Reliability Diagram — {split_name} (seed {seed})")
    ax1.legend()

    ax2.hist(max_probs, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Max predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Distribution")

    plt.tight_layout()
    if save_dir:
        fig_path = save_dir / f"calibration_{split_name}_seed{seed}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()


def run_calibration_analysis(result: Dict) -> Dict:
    \"\"\"Compute ECE and Brier score for a completed training run.\"\"\"
    seed      = result["seed"]
    prob_cols = [f"prob_{lbl}" for lbl in LABEL_ORDER]

    val_df    = result["val_pred_df"]
    val_true  = [LABEL2ID[lbl] for lbl in val_df["true_label"]]
    val_probs = val_df[prob_cols].values

    test_df    = result["test_pred_df"]
    test_true  = [LABEL2ID[lbl] for lbl in test_df["true_label"]]
    test_probs = test_df[prob_cols].values

    val_ece    = compute_ece(val_true,  val_probs,  n_bins=15)
    val_brier  = compute_brier_score(val_true,  val_probs,  NUM_LABELS)
    test_ece   = compute_ece(test_true, test_probs, n_bins=15)
    test_brier = compute_brier_score(test_true, test_probs, NUM_LABELS)

    print(f"\\n── Calibration — Seed {seed} ──")
    print(f"  Val  ECE   : {val_ece:.4f}  (lower = better)")
    print(f"  Val  Brier : {val_brier:.4f}")
    print(f"  Test ECE   : {test_ece:.4f}")
    print(f"  Test Brier : {test_brier:.4f}")

    calibration_plot(val_true,  val_probs,  "validation", seed, save_dir=DIRS["out_figures"])
    calibration_plot(test_true, test_probs, "test",       seed, save_dir=DIRS["out_figures"])

    # Save calibration metrics
    cal_path = DIRS["out_metrics"] / f"calibration_seed{seed}.json"
    with open(cal_path, "w") as f:
        json.dump({"seed": seed, "val_ece": val_ece, "val_brier": val_brier,
                   "test_ece": test_ece, "test_brier": test_brier}, f, indent=2)

    return {"seed": seed, "val_ece": val_ece, "val_brier": val_brier,
            "test_ece": test_ece, "test_brier": test_brier}


print("✅  Calibration functions defined.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 17 — Error Analysis
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 17 — Error analysis

Error analysis helps us understand **where and why the model fails**, not just how often.

### High-confidence errors

These are the most informative failure cases — the model was very certain (e.g., 95% \
confidence) but wrong. They reveal **systematic model biases**, not random mistakes.

### Expected confusion patterns

- Anger ↔ Disgust: both are negative, aversive emotions — often linguistically similar
- Joy ↔ Surprise: excited, positive text can be ambiguous
- Sadness ↔ Neutral: low-affect sad text may be hard to distinguish from neutral

Unexpected confusions (e.g., Sadness predicted as Joy) indicate a more serious problem \
and should be investigated by reading the actual examples.

### Why we display examples

Automated confusion matrices alone can miss important patterns. Manual review of \
high-confidence errors is a core part of responsible NLP research.
"""))

C.append(code("""\
def plot_confusion_matrix(
    true_labels : List[int],
    pred_labels : List[int],
    label_names : List[str],
    seed        : int,
    split       : str = "test",
    save_dir    : Optional[Path] = None,
) -> None:
    \"\"\"Plot raw-count and row-normalised confusion matrices side-by-side.\"\"\"
    cm      = confusion_matrix(true_labels, pred_labels, labels=list(range(len(label_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    kw = dict(xticklabels=label_names, yticklabels=label_names)

    sns.heatmap(cm,      annot=True, fmt="d",    cmap="Blues", ax=ax1, **kw)
    ax1.set_title(f"Confusion Matrix — Raw Counts\\n{split} split (seed {seed})")
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f",  cmap="Blues", ax=ax2, **kw)
    ax2.set_title(f"Confusion Matrix — Row-Normalised (Recall)\\n{split} split (seed {seed})")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")

    plt.tight_layout()
    if save_dir:
        fig_path = save_dir / f"confusion_matrix_{split}_seed{seed}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()


def run_error_analysis(
    result     : Dict,
    df         : pd.DataFrame,
    n_examples : int = 10,
    split      : str = "test",
) -> None:
    \"\"\"Display confusion matrix, per-class report, and example errors.\"\"\"
    seed      = result["seed"]
    pred_df   = result[f"{split}_pred_df"].copy()
    prob_cols = [f"prob_{lbl}" for lbl in LABEL_ORDER]

    true_int = [LABEL2ID[lbl] for lbl in pred_df["true_label"]]
    pred_int = [LABEL2ID[lbl] for lbl in pred_df["pred_label"]]

    plot_confusion_matrix(true_int, pred_int, LABEL_ORDER, seed=seed,
                          split=split, save_dir=DIRS["out_figures"])

    print(f"\\n── Per-class report ({split}, seed {seed}) ──")
    print(classification_report(true_int, pred_int, target_names=LABEL_ORDER,
                                 labels=list(range(NUM_LABELS)), zero_division=0))

    pred_df["max_prob"]     = pred_df[prob_cols].max(axis=1)
    id_to_text              = dict(zip(df["example_id"], df["text"]))
    pred_df["original_text"] = pred_df["example_id"].map(id_to_text)

    errors  = pred_df[~pred_df["correct"]].copy()
    correct = pred_df[pred_df["correct"]].copy()

    print(f"\\n── Error summary ({split}, seed {seed}) ──")
    print(f"  Total     : {len(pred_df):,}")
    print(f"  Correct   : {len(correct):,}  ({len(correct)/len(pred_df)*100:.1f}%)")
    print(f"  Incorrect : {len(errors):,}  ({len(errors)/len(pred_df)*100:.1f}%)")

    print(f"\\n── Top {n_examples} highest-confidence ERRORS ──")
    for _, row in errors.nlargest(n_examples, "max_prob").iterrows():
        txt = str(row["original_text"])[:80]
        print(f"  True: {row['true_label']:<10} Pred: {row['pred_label']:<10} "
              f"Conf: {row['max_prob']:.3f}  │  {txt!r}")

    print(f"\\n── Top {n_examples} lowest-confidence CORRECT predictions ──")
    for _, row in correct.nsmallest(n_examples, "max_prob").iterrows():
        txt = str(row["original_text"])[:80]
        print(f"  True: {row['true_label']:<10} Pred: {row['pred_label']:<10} "
              f"Conf: {row['max_prob']:.3f}  │  {txt!r}")

    err_path = DIRS["out_predictions"] / f"errors_{split}_seed{seed}.csv"
    display_cols = ["example_id","true_label","pred_label","correct","max_prob"] + prob_cols
    errors[display_cols].to_csv(err_path, index=False)
    print(f"\\nError CSV saved: {err_path}")


print("✅  Error analysis functions defined.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 18 — Results Aggregation
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 18 — Results aggregation and training curves

After running all seeds, we aggregate results into a summary table with mean and \
standard deviation.

### Why mean ± std?

A single run may be unusually good or bad due to weight initialisation, data-loading \
order, and other stochastic effects. Three seeds give an **initial estimate** of \
run-to-run variability. Standard deviation quantifies this.

$$\\text{Mean} = \\frac{1}{3} \\sum_{i=1}^{3} S_i$$

Three seeds are a practical starting point. For publication-quality results, more seeds \
or a held-out validation approach (e.g., 5-fold CV) would be appropriate.

> Do not average raw logits from different seeds to produce ensemble predictions unless \
a separate ensemble experiment is explicitly designed and documented.
"""))

C.append(code("""\
def aggregate_results(
    all_results : List[Dict],
    all_calib   : List[Dict],
) -> pd.DataFrame:
    \"\"\"Build a summary DataFrame from multi-seed results.\"\"\"
    rows = []
    for res, cal in zip(all_results, all_calib):
        rows.append({
            "seed"              : res["seed"],
            "best_epoch"        : res["best_epoch"],
            "best_val_macro_f1" : res["best_val_macro_f1"],
            "test_accuracy"     : res["test_metrics"]["test_accuracy"],
            "test_macro_f1"     : res["test_metrics"]["test_macro_f1"],
            "test_weighted_f1"  : res["test_metrics"]["test_weighted_f1"],
            "val_ece"           : cal["val_ece"],
            "val_brier"         : cal["val_brier"],
            "test_ece"          : cal["test_ece"],
            "test_brier"        : cal["test_brier"],
        })

    df_sum = pd.DataFrame(rows)
    num_cols = [c for c in df_sum.columns if c not in ("seed", "best_epoch")]
    mean_row = {"seed": "mean", "best_epoch": ""}
    std_row  = {"seed": "std",  "best_epoch": ""}
    for col in num_cols:
        mean_row[col] = df_sum[col].mean()
        std_row[col]  = df_sum[col].std(ddof=1)
    df_sum = pd.concat([df_sum, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    return df_sum


def plot_training_curves(all_results: List[Dict]) -> None:
    \"\"\"Plot loss and macro-F1 curves for all seeds.\"\"\"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["steelblue", "darkorange", "forestgreen"]

    for color, res in zip(colors, all_results):
        h       = res["history"]
        epochs  = [r["epoch"]          for r in h]
        tl      = [r["train_loss"]     for r in h]
        vl      = [r["val_loss"]       for r in h]
        vm      = [r["val_macro_f1"]   for r in h]
        seed    = res["seed"]
        ax1.plot(epochs, tl, "--", color=color, alpha=0.6, label=f"train s{seed}")
        ax1.plot(epochs, vl, "-",  color=color,            label=f"val   s{seed}")
        ax2.plot(epochs, vm, "-o", color=color,            label=f"seed {seed}")

    ax1.set_title("Loss Curves (all seeds)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(fontsize=8)
    ax2.set_title("Validation Macro-F1 (all seeds)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro-F1"); ax2.legend()

    plt.suptitle(
        f"Training History — {CFG['experiment_name']}  (Condition {CFG['condition_label']})",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = DIRS["out_figures"] / "training_curves_all_seeds.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved: {fig_path}")


print("✅  Aggregation and plotting functions defined.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 19 — Single-Seed Debug Run
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 19 — Single-seed run (pipeline verification)

> ⚠️ **Run this section first.** Complete one full training run with seed 42 before \
attempting the three-seed loop. This catches data-loading errors, shape mismatches, \
and OOM errors early, without wasting GPU time on a broken pipeline.

### If you encounter a CUDA out-of-memory error:

1. Reduce `CFG["train_batch_size"]` from 16 → 8
2. Increase `CFG["grad_accum_steps"]` from 1 → 2 (keeps effective batch = 16)
3. Reduce `CFG["max_seq_len"]` from 128 → 96
4. **Restart the runtime** (Runtime → Restart runtime) and re-run from the top

Document any deviation from the planned configuration. Do not silently change settings \
mid-run.

### Expected output of a successful single-seed run

- Checkpoint saved to `models/english_emotion/seed_42/`
- Validation logits CSV in `outputs/english_emotion/logits/`
- Test logits CSV in `outputs/english_emotion/logits/`
- Training history JSON in `outputs/english_emotion/logs/`
- Calibration plots in `outputs/english_emotion/figures/`
- Confusion matrix in `outputs/english_emotion/figures/`
"""))

C.append(code("""\
# ── Set this flag to True for first run; False after pipeline is verified ──
SINGLE_SEED_DEBUG = True   # <-- change to False before running Section 20

if SINGLE_SEED_DEBUG:
    print("Pipeline verification mode: running seed 42 only.")
    print("After this succeeds, set SINGLE_SEED_DEBUG = False and run Section 20.\\n")

    result_debug  = run_training(
        seed          = 42,
        cfg           = CFG,
        df            = df,
        tokenizer     = tokenizer,
        class_weights = CLASS_WEIGHTS_TENSOR if CFG["use_class_weights"] else None,
    )
    calib_debug   = run_calibration_analysis(result_debug)
    run_error_analysis(result_debug, df, n_examples=10, split="test")

    print("\\n" + "="*60)
    print("SINGLE-SEED VERIFICATION COMPLETE")
    print("="*60)
    print(f"  Test macro-F1  : {result_debug['test_metrics']['test_macro_f1']:.4f}")
    print(f"  Test weighted-F1: {result_debug['test_metrics']['test_weighted_f1']:.4f}")
    print(f"  Test ECE       : {calib_debug['test_ece']:.4f}")
    print(f"  Best epoch     : {result_debug['best_epoch']}")
    print("\\nIf all outputs look correct, set SINGLE_SEED_DEBUG = False")
    print("in this cell and run Section 20 (multi-seed).")
else:
    print("SINGLE_SEED_DEBUG is False — skipping. Proceed to Section 20.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 20 — Multi-Seed Experiment
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 20 — Multi-seed experiment

After the single-seed run succeeds and you have set `SINGLE_SEED_DEBUG = False` in \
Section 19, run this cell to execute all three seeds sequentially.

### Time estimate (T4 GPU, 3 epochs per seed)

| Seeds | Approximate total time |
|---|---|
| 3 seeds × ~40 min/seed | ~2 hours |

> 💡 Colab free-tier runtimes may disconnect after ~90 minutes of inactivity. \
Connect to a GPU runtime with enough time remaining, or use the `/goal` command \
to run the full experiment reliably.

### What gets saved per seed

- Best model checkpoint + tokeniser + config
- Validation logits (best epoch)
- Test logits
- Training history JSON
- Calibration plots
- Confusion matrix + error CSV
"""))

C.append(code("""\
# ── Linked to the flag in Section 19 ─────────────────────────────────────
RUN_MULTI_SEED = not SINGLE_SEED_DEBUG

all_results = []
all_calib   = []

if RUN_MULTI_SEED:
    for seed in CFG["seeds"]:
        result = run_training(
            seed          = seed,
            cfg           = CFG,
            df            = df,
            tokenizer     = tokenizer,
            class_weights = CLASS_WEIGHTS_TENSOR if CFG["use_class_weights"] else None,
        )
        calib  = run_calibration_analysis(result)
        run_error_analysis(result, df, n_examples=5, split="test")

        all_results.append(result)
        all_calib.append(calib)

    # ── Summary table ─────────────────────────────────────────────────────
    summary_df = aggregate_results(all_results, all_calib)
    print("\\n" + "=" * 70)
    print("MULTI-SEED RESULTS SUMMARY")
    print("=" * 70)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_columns", 15, "display.width", 120):
        print(summary_df.to_string(index=False))

    summary_path = DIRS["out_metrics"] / "multi_seed_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\\nSummary saved: {summary_path}")

    plot_training_curves(all_results)

else:
    print("Multi-seed run skipped (SINGLE_SEED_DEBUG = True in Section 19).")
    print("Set SINGLE_SEED_DEBUG = False and re-run this cell.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 21 — Final Metadata & Summary
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 21 — Final experiment summary and metadata

This cell writes the definitive experiment metadata file and prints a human-readable \
summary of everything saved. Run this cell after either the single-seed or multi-seed \
experiment completes.
"""))

C.append(code("""\
final_meta = {
    "experiment_name"   : CFG["experiment_name"],
    "condition"         : CFG["condition_label"],
    "backbone"          : CFG["model_name"],
    "task_data"         : "GoEmotions-Ekman7 (English only)",
    "label_order"       : LABEL_ORDER,
    "id2label"          : ID2LABEL,
    "label2id"          : LABEL2ID,
    "dataset_path"      : CFG["dataset_path"],
    "dataset_counts"    : {
        s: int((df["split"] == s).sum()) for s in ["train", "validation", "test"]
    },
    "hyperparameters"   : {k: str(v) for k, v in CFG.items() if k != "seeds"},
    "seeds"             : CFG["seeds"],
    "primary_metric"    : CFG["primary_metric"],
    "software"          : {
        "python"       : sys.version,
        "pytorch"      : torch.__version__,
        "transformers" : transformers.__version__,
        "numpy"        : np.__version__,
        "pandas"       : pd.__version__,
    },
    "completed_at"      : datetime.datetime.now().isoformat(),
    "research_notes"    : (
        "Condition A: XLM-R backbone, English-only GoEmotions training. "
        "Class-weighted CrossEntropyLoss (balanced). "
        "Checkpoint selection by validation macro-F1. "
        "Test set evaluated exactly once per seed after model selection. "
        "Three independent seeds to quantify variability."
    ),
}

meta_path = DIRS["out_metrics"] / "experiment_metadata.json"
with open(meta_path, "w") as f:
    json.dump(final_meta, f, indent=2)

print("=" * 60)
print("EXPERIMENT COMPLETE — BEYOND WORDS CONDITION A")
print("=" * 60)
print(f"  Backbone      : {CFG['model_name']}")
print(f"  Task data     : English GoEmotions Ekman-7")
print(f"  Label order   : {LABEL_ORDER}")
print(f"  Seeds planned : {CFG['seeds']}")
print(f"\\nOutput directories:")
for name, path in DIRS.items():
    if "out_" in name or name == "models_root":
        print(f"  {name:<20} : {path}")
print(f"\\nMetadata       : {meta_path}")
print(f"\\nNext steps:")
print(f"  Condition B : XLM-R + English + Hinglish task data")
print(f"  Condition C : HingRoBERTa + English only")
print(f"  Condition D : HingRoBERTa + English + Hinglish")
print(f"  Zero-shot   : evaluate Condition A on HinEmo-9 and EmoSen (Notebook 05)")
"""))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 22 — Verification Cell
# ════════════════════════════════════════════════════════════════════════════
C.append(md("""\
## Section 22 — Output verification

Run this cell after training to confirm all expected outputs are present and non-empty.
"""))

C.append(code("""\
# ── Verification: confirm all expected outputs are present ─────────────────
VERIFY_SEED = 42   # adjust to match the seed you ran

def check_exists(label: str, path) -> bool:
    p = Path(path) if path else None
    if p is None:
        print(f"  ❌  {label:<35} : path is None")
        return False
    if p.exists() and p.stat().st_size > 0:
        print(f"  ✅  {label:<35} : {p.stat().st_size / 1024:.1f} KB")
        return True
    elif p.exists():
        print(f"  ⚠️   {label:<35} : exists but EMPTY")
        return False
    else:
        print(f"  ❌  {label:<35} : MISSING — {p}")
        return False

print("=" * 60)
print(f"OUTPUT VERIFICATION (seed {VERIFY_SEED})")
print("=" * 60)

seed_dir = DIRS["models_root"] / f"seed_{VERIFY_SEED}"

checks = {
    "Config JSON"            : DIRS["configs"] / "english_emotion_config.json",
    "Model config.json"      : seed_dir / "config.json",
    "Model pytorch_model.bin": seed_dir / "pytorch_model.bin",
    "Run metadata"           : seed_dir / "run_metadata.json",
    "Training history"       : DIRS["out_logs"]   / f"training_history_seed{VERIFY_SEED}.json",
    "Test logits CSV"        : DIRS["out_logits"]  / f"test_logits_seed{VERIFY_SEED}.csv",
    "Experiment metadata"    : DIRS["out_metrics"] / "experiment_metadata.json",
    "Class dist figure"      : DIRS["out_figures"] / "class_distribution.png",
    "Token len figure"       : DIRS["out_figures"] / "token_length_distribution.png",
    "Calibration test fig"   : DIRS["out_figures"] / f"calibration_test_seed{VERIFY_SEED}.png",
    "Confusion matrix"       : DIRS["out_figures"] / f"confusion_matrix_test_seed{VERIFY_SEED}.png",
}

# Val logits filename includes epoch number; find with glob
val_logit_matches = list(DIRS["out_logits"].glob(f"val_logits_seed{VERIFY_SEED}_*.csv"))
checks["Val logits CSV (best epoch)"] = val_logit_matches[0] if val_logit_matches else None

all_ok = all(check_exists(lbl, path) for lbl, path in checks.items())
print()
if all_ok:
    print("✅  All expected outputs verified successfully.")
    print("   The pipeline completed cleanly for seed", VERIFY_SEED)
else:
    print("⚠️  Some outputs are missing. Review errors above.")
    print("   Common causes: training interrupted, OOM error, or Section 19 did not complete.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# ASSEMBLE AND WRITE NOTEBOOK
# ════════════════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
        "accelerator": "GPU",
    },
    "cells": C,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, indent=1, ensure_ascii=False)

print(f"Written: {OUT}")
print(f"Cells  : {len(C)}")
print("Done.")
