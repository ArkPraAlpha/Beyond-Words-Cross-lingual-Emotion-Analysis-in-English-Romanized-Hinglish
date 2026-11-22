# Beyond Words — Roadmap

> This file tracks planned work, known gaps, and future research directions.

---

## 🔴 High Priority

### Sarcasm Detection Module
**Status:** Not started  
**Why:** Sarcasm inverts emotional valence. A sarcastic "Oh great, delayed again!" reads as anger, not joy. Without sarcasm awareness, the classifier will systematically misclassify emotionally ironic Hinglish social-media content.

**Planned approach:**
- [ ] Survey available Hinglish sarcasm datasets (SemEval, iSarcasm, any Hindi code-switched resources)
- [ ] Collect/annotate a Hinglish sarcasm corpus if none exists (target: 2,000+ examples)
- [ ] Train a binary sarcasm detector (or joint emotion–sarcasm model)
- [ ] Add Notebook 08: `08_sarcasm_detection.ipynb`
- [ ] Integrate into the app as an optional pre-classification step
- [ ] Update `app/model_cards.json` with sarcasm model card

**New experimental conditions to add:**
| Condition | Backbone | Data | Notes |
|---|---|---|---|
| E | HingRoBERTa | Sarcasm corpus | Binary sarcasm classifier |
| F | HingRoBERTa | Emotion + Sarcasm joint | Multi-task head |

---

### Probability Recalibration
**Status:** Measured, not fixed  
**Problem:** Raw softmax probabilities are overconfident. ECE reaches 0.262 on HinEmo-9 for Condition A.

**Planned approach:**
- [ ] Fit temperature scaling T* on the validation set for each condition
- [ ] Add `08_calibration.ipynb` or extend NB 06 with calibration fitting
- [ ] Replace app temperature slider default (currently 1.0) with the fitted T*
- [ ] Re-report calibration metrics after recalibration

---

## 🟡 Medium Priority

### Larger Hinglish Benchmarks
**Status:** Identified gap  
**Problem:** HinEmo-9 (270 examples, 5 classes) is too small for reliable statistical comparison.

- [ ] Find or create a larger Hinglish emotion test set (target: >3,000 examples, 7 classes)
- [ ] Include YouTube, Twitter, Instagram, and WhatsApp-style data

### Statistical Significance Testing
**Status:** Not done  
**Problem:** Reporting mean ± SD over 3 seeds does not constitute a formal statistical test.

- [ ] Run McNemar's test or bootstrap confidence intervals on condition-pair differences
- [ ] Add statistical significance annotations to Notebook 06 results

### Sentiment-Intent Module
**Status:** Not started (original project vision)  
**Goal:** Beyond emotion labels, classify the communicative intent  
(complaint, praise, question, sarcasm, humour, neutral statement)

- [ ] Design annotation schema for intent categories
- [ ] Add Notebook 09: `09_sentiment_intent.ipynb`

---

## 🟢 Nice to Have

### Production Deployment
**Status:** Colab + ngrok (not production-ready)  
**Goal:** A persistent, always-on endpoint that anyone can use.

- [ ] Deploy FastAPI server to HuggingFace Spaces or Render.com
- [ ] Add proper rate limiting and authentication
- [ ] Remove Colab dependency from the app

### Multi-label Output Mode
**Status:** Not started  
**Goal:** Return top-k emotions with individual probabilities instead of forcing a single label.

### HuggingFace Hub Upload
**Status:** Not started  
**Goal:** Upload trained checkpoints to HuggingFace Hub so anyone can download them.

- [ ] Create model cards for all 4 conditions on HF Hub
- [ ] Link from README

### Condition B & C Outputs
**Status:** Computed on Colab but not pushed to this repo  
**Goal:** Populate `outputs/condB/` and `outputs/condC/` with figures, logits, metrics.

- [ ] Run Notebook 04 (Condition C) → copy outputs to `outputs/condC/`
- [ ] Run Notebook 05 (Condition B) → copy outputs to `outputs/condB/`
- [ ] Push updated outputs to GitHub

---

## Completed ✅

- [x] GoEmotions preprocessing pipeline (Notebook 01)
- [x] Hinglish dataset audit — EmoSen + HinEmo-9 (Notebook 02)
- [x] Condition A: XLM-R English-only baseline — 3 seeds (Notebook 03)
- [x] Condition C: HingRoBERTa English-only — 3 seeds (Notebook 04)
- [x] Conditions B & D: Hinglish augmentation — 3 seeds each (Notebook 05)
- [x] Cross-condition evaluation A–D (Notebook 06)
- [x] EmoSen anger→disgust shift diagnostic (Notebook 07)
- [x] FastAPI Colab inference server (`colab/bw_api.py`)
- [x] Streamlit local app with Light/Dark theme, all 4 conditions (`app/app.py`)
- [x] Smoke test for prediction reproducibility (`scripts/smoke_test.py`)
- [x] ECE and Brier score measurement (reported; recalibration pending)
- [x] Full project PDF report and literature review
