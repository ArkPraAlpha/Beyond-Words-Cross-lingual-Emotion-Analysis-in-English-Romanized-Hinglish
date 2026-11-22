# Beyond Words
### Cross-lingual Emotion Analysis in English & Romanized Hinglish

> **Research prototype** — emotion classification across English and romanized Hinglish using a 2×2 factorial transformer experiment.  
> **Status:** Emotion module complete (Conditions A–D). Sarcasm module planned — see [ROADMAP.md](ROADMAP.md).

---

## What is this?

**Beyond Words** is an NLP research project that fine-tunes multilingual transformer models to detect emotions in both **English** and **Romanized Hinglish** — the mixed Hindi–English code-switched text that dominates South Asian social media.

It addresses a well-documented gap: standard English emotion classifiers fail on Hinglish because they can't handle code-switching, non-standard spelling, and social-media linguistic norms.

### Seven emotion classes (Ekman mapping)
`anger` · `disgust` · `fear` · `joy` · `sadness` · `surprise` · `neutral`

---

## Experimental Design — 2×2 Factorial

|   | English-only training | + Hinglish augmentation (EmoSen) |
|---|---|---|
| **XLM-RoBERTa** (base) | **Condition A** ← baseline | **Condition B** |
| **HingRoBERTa** (Hinglish-adapted XLM-R) | **Condition C** | **Condition D** ⭐ recommended |

Each condition is trained with **3 independent random seeds** (42, 123, 2026) for variability estimation. The test set is evaluated **exactly once** per seed after model selection.

---

## Key Results

| Condition | Backbone | GoEmotions Macro-F1 | EmoSen Macro-F1 | HinEmo-9 Macro-F1 |
|---|---|---|---|---|
| A — XLM-R, English only | XLM-R-base | 0.722 ± 0.011 | 0.280 ± 0.006 | 0.374 ± 0.013 |
| B — XLM-R + EmoSen | XLM-R-base | 0.726 ± 0.001 | 0.287 ± 0.016 | 0.360 ± 0.021 |
| C — HingRoBERTa, English only | HingRoBERTa | 0.733 ± 0.002 | 0.353 ± 0.025 | 0.493 ± 0.021 |
| **D — HingRoBERTa + EmoSen** ⭐ | **HingRoBERTa** | **0.726 ± 0.005** | **0.368 ± 0.019** | **0.527 ± 0.006** |

> Scores are means ± SD over 3 seeds. HinEmo-9 test set is small (n=270); differences between conditions should not be over-interpreted.

---

## Project Structure

```
beyond-words/
│
├── notebooks/                   ← 7 Colab notebooks (run in order)
│   ├── 01_dataset_exploration.ipynb           Phase 1: GoEmotions EDA & Ekman mapping
│   ├── 02_hinglish_dataset_audit.ipynb        Phase 1: EmoSen + HinEmo-9 audit
│   ├── 03_english_emotion_baseline.ipynb      Condition A: XLM-R, English only
│   ├── 04_english_emotion_hingroberta_condC.ipynb  Condition C: HingRoBERTa, English only
│   ├── 05_hinglish_adaptation_condB_condD.ipynb    Conditions B & D: Hinglish augmentation
│   ├── 06_hinglish_eval_ABCD_comparison.ipynb      Cross-condition evaluation
│   └── 07_emosen_split_shift_diagnostic.ipynb      EmoSen anger→disgust shift analysis
│
├── app/                         ← Streamlit local UI (connects to Colab API)
│   ├── app.py                   Full production UI (Light/Dark theme, all conditions)
│   ├── bw_utils.py              Shared utilities (softmax, confidence, checkpoint discovery)
│   ├── model_cards.json         Evaluation summaries for all 4 conditions
│   ├── requirements.txt         pip dependencies for the local app
│   └── README.md                How to run the app
│
├── colab/                       ← Colab inference server
│   ├── Beyond_Words_Colab_Server.ipynb   Start the FastAPI server on Colab GPU
│   └── bw_api.py                FastAPI inference service (Bearer-auth, one checkpoint at a time)
│
├── configs/                     ← Frozen experiment hyperparameters
│   ├── english_emotion_config.json         Condition A config
│   └── hinglish_adapt_condD_config.json    Condition D config
│
├── outputs/                     ← All experiment outputs (tracked in git)
│   ├── condA/                   Condition A: XLM-R English-only
│   │   ├── figures/             Confusion matrices, calibration plots, training curves
│   │   ├── logits/              Raw logit CSVs (GoEmotions, EmoSen, HinEmo-9 test sets)
│   │   ├── logs/                Per-epoch training history JSON
│   │   ├── metrics/             Calibration, Brier, ECE, multi-seed summary
│   │   └── predictions/         Error analysis CSVs
│   ├── condB/                   Condition B outputs (run NB 05 to generate → push here)
│   ├── condC/                   Condition C outputs (run NB 04/06 to generate → push here)
│   ├── condD/                   Condition D: HingRoBERTa + EmoSen ✅
│   │   ├── figures/             Training curves
│   │   ├── logits/              Raw logit CSVs across all test sets
│   │   ├── logs/                Training history JSON
│   │   └── metrics/             Multi-test-set evaluation summary
│   └── cross_condition/         Shared metrics: ABCD comparison CSVs, condC logits
│
├── scripts/                     ← Utility scripts
│   ├── smoke_test.py            Reproducibility check: app predictions vs saved logits
│   └── build_nb.py              Programmatic notebook builder (generates NB 03)
│
├── data/                        ← Data directory (files excluded from git — see below)
│   └── processed/
│       └── .gitkeep             Placeholder; download goemotions_ekman_dedup.csv from Drive
│
├── models/                      ← Model checkpoints (weights excluded from git)
│   └── .gitkeep                 Weights stored on Google Drive / HuggingFace Hub
│
├── reports/                     ← PDF documentation
│   ├── project_report_final.pdf          Full 19-page technical project report
│   ├── literature_review_and_gap_analysis.pdf  Literature review & research gaps
│   └── learning_guide.pdf               Plain-language project explainer
│
├── .gitignore
├── README.md                    ← You are here
└── ROADMAP.md                   ← Planned features (sarcasm, calibration, etc.)
```

---

## Quickstart — Running the App

> **Requirements:** Python 3.10 or 3.11 locally. GPU not needed on your machine.

### Step 1 — Start Colab inference server
1. Open `colab/Beyond_Words_Colab_Server.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set runtime to **T4 GPU** (Runtime → Change runtime type).
3. Run all cells. Enter your [ngrok authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) when prompted.
4. Copy the **HTTPS API URL** and **API key** printed at the end.

### Step 2 — Run the local Streamlit app
```powershell
cd app
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
Then paste the Colab URL and API key into the sidebar.

> **macOS / Linux:** replace `Activate.ps1` with `source .venv/bin/activate`

### Verify reproducibility (optional)
```powershell
python scripts/smoke_test.py ^
  models/english_emotion/xlmr_english_ekman7_condA/seed_42 ^
  outputs/condA/logits/test_logits_xlmr_english_ekman7_condA_seed42.csv ^
  data/processed/goemotions_ekman_dedup.csv
# Should print: PASS
```

---

## Datasets

| Dataset | Source | Used for | In repo? |
|---|---|---|---|
| GoEmotions (Ekman-7) | [Google Research](https://github.com/google-research/google-research/tree/master/goemotions) | Training (Conditions A–D) | ❌ Too large (~10 MB) — download from Drive |
| EmoSen | [Ghosh et al.](https://github.com/SAbhijit30/EmoSen) | Augmentation (B, D) + eval | ❌ Excluded — download from Drive |
| HinEmo-9 | [Pal et al.] | Hinglish eval | ❌ Excluded — handled by notebooks |

> **Data download:** All processed datasets are on the project Google Drive. Run notebooks 01 and 02 to regenerate from scratch, or copy `goemotions_ekman_dedup.csv` into `data/processed/`.

---

## Model Weights

Trained model weights are **not stored in this repository** (each checkpoint is ~1 GB).

| Condition | Model ID (HuggingFace Hub / Google Drive) |
|---|---|
| A — XLM-R, English | Google Drive: `models/english_emotion/xlmr_english_ekman7_condA/seed_{42,123,2026}` |
| B — XLM-R + EmoSen | Google Drive: `outputs/hinglish_adapt_condB/...` |
| C — HingRoBERTa, English | Google Drive: `outputs/hinglish_adapt_condC/...` |
| D — HingRoBERTa + EmoSen | Google Drive: `outputs/hinglish_adapt_condD/...` |

To use a checkpoint locally, copy its folder (must contain `config.json`, `model.safetensors`, and tokenizer files) into `models/`.

---

## Notebooks Guide

| # | Notebook | Condition | Outputs |
|---|---|---|---|
| 01 | `01_dataset_exploration.ipynb` | — | `data/processed/goemotions_ekman_dedup.csv`, audit files |
| 02 | `02_hinglish_dataset_audit.ipynb` | — | EmoSen/HinEmo-9 audit reports |
| 03 | `03_english_emotion_baseline.ipynb` | **A** | `outputs/condA/` (complete ✅) |
| 04 | `04_english_emotion_hingroberta_condC.ipynb` | **C** | `outputs/condC/` (run to populate) |
| 05 | `05_hinglish_adaptation_condB_condD.ipynb` | **B, D** | `outputs/condB/`, `outputs/condD/` |
| 06 | `06_hinglish_eval_ABCD_comparison.ipynb` | A–D | `outputs/cross_condition/` |
| 07 | `07_emosen_split_shift_diagnostic.ipynb` | B, D | Anger→disgust shift analysis |

---

## Limitations

- **No sarcasm detection** — Sarcastic text can invert emotional valence. See [ROADMAP.md](ROADMAP.md).
- **Small Hinglish test sets** — HinEmo-9 (n=270) and EmoSen test (n=1,927, severely imbalanced).
- **Uncalibrated probabilities** — Raw softmax is overconfident (ECE up to 0.262 on HinEmo-9).
- **Anger→Disgust shift** — Conditions B & D show a systematic label shift on EmoSen under investigation.
- **Colab dependency** — The live app requires an active Colab runtime (not production-ready).
- Emotion labels reflect annotation conventions, not a person's true internal state.  
  **Do not use this output to make decisions about people.**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Pre-trained models | `FacebookAI/xlm-roberta-base`, `l3cube-pune/hing-roberta` |
| Training | PyTorch 2.15 · HuggingFace Transformers 4.48 · Google Colab T4 GPU |
| Inference server | FastAPI + uvicorn + ngrok tunnel |
| UI | Streamlit ≥ 1.40 |
| Data | GoEmotions · EmoSen · HinEmo-9 |

---

## Reports

| Document | Description |
|---|---|
| [`reports/project_report_final.pdf`](reports/project_report_final.pdf) | Full 19-page technical report |
| [`reports/literature_review_and_gap_analysis.pdf`](reports/literature_review_and_gap_analysis.pdf) | Literature review & research gap analysis |
| [`reports/learning_guide.pdf`](reports/learning_guide.pdf) | Plain-language project explainer |

---

## Citation

If you use this code or results in your work, please cite:

```bibtex
@misc{beyondwords2026,
  title   = {Beyond Words: Cross-lingual Emotion Analysis in English and Romanized Hinglish},
  year    = {2026},
  url     = {https://github.com/YOUR_USERNAME/beyond-words},
  note    = {Research prototype. Emotion-only; sarcasm module planned.}
}
```

---

## License

This project is released for research and educational purposes.  
Pre-trained model weights are subject to their respective upstream licenses:
- XLM-RoBERTa: [MIT License](https://huggingface.co/FacebookAI/xlm-roberta-base)
- HingRoBERTa: [MIT License](https://huggingface.co/l3cube-pune/hing-roberta)
- GoEmotions dataset: [Apache 2.0](https://github.com/google-research/google-research/tree/master/goemotions)
