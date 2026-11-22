"""Beyond Words: emotion demo. Run with:  streamlit run app/app.py
Uses the checkpoints you trained under models/. Emotion only: there is no sarcasm or
sentiment-intent model in this prototype, and the app does not pretend to have one."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from bw_utils import (LABEL_ORDER, MAX_LEN, confidence_summary, discover_checkpoints,
                      load_cards, parse_condition, softmax_np)

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BEYOND_WORDS_ROOT", HERE.parent))
CARDS = load_cards(HERE / "model_cards.json")

st.set_page_config(page_title="Beyond Words", page_icon="💬", layout="centered")
st.title("Beyond Words")
st.caption("Emotion classifier for English and romanized Hinglish text. Research prototype, "
           "emotion only (no sarcasm or intent analysis).")

checkpoints = discover_checkpoints(ROOT / "models")
if not checkpoints:
    st.error(f"No checkpoints found under {ROOT / 'models'}. Copy a trained checkpoint folder "
             "(config.json, model.safetensors, tokenizer files) there, or set BEYOND_WORDS_ROOT.")
    st.stop()


def label_for(path: Path) -> str:
    cond, seed = parse_condition(path)
    meta = CARDS.get("conditions", {}).get(cond or "", {})
    name = meta.get("name", path.parent.name)
    return f"Condition {cond}: {name} (seed {seed})" if cond else str(path.relative_to(ROOT))


options = {label_for(p): p for p in checkpoints}
choice = st.sidebar.selectbox("Model checkpoint", list(options))
temperature = st.sidebar.slider("Softmax temperature", 0.5, 3.0, 1.0, 0.1,
                                help="1.0 = raw model probabilities. No temperature has been "
                                     "fitted yet, so leave at 1.0 unless you are exploring.")
ckpt = options[choice]


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    if model.config.num_labels != len(LABEL_ORDER):
        raise ValueError(f"Expected {len(LABEL_ORDER)} labels, checkpoint has {model.config.num_labels}.")
    model.eval()
    return tok, model


def predict_logits(text: str, tok, model):
    import torch
    enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    n_tokens = int(enc["input_ids"].shape[1])
    with torch.inference_mode():
        logits = model(**enc).logits[0].float().cpu().numpy()
    return logits, n_tokens


EXAMPLES = [
    "I can't believe they cancelled my flight again, this is so frustrating.",
    "Aaj bahut accha din tha, sab dost mil gaye!",
    "Mujhe bahut dar lag raha hai kal ke result se.",
    "Yaar kal ki match dekhi? Bilkul unexpected tha!",
]
st.write("Type a sentence, or pick an example (examples only fill the box; the prediction is always "
         "computed live by the selected checkpoint).")
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(f"Example {i + 1}"):
        st.session_state["text"] = ex
text = st.text_area("Text", key="text", height=100, placeholder="Enter English or romanized Hinglish text")

if st.button("Analyse", type="primary") and text.strip():
    tok, model = load_model(str(ckpt))
    logits, n_tokens = predict_logits(text.strip(), tok, model)
    probs = softmax_np(logits, temperature)
    s = confidence_summary(probs)
    st.subheader(f"Predicted emotion: {s['label']}  ({s['prob']:.0%})")
    if s["flag"] == "uncertain":
        st.warning(f"The model is not sure: runner-up is {s['runner_up']} ({s['runner_up_prob']:.0%}). "
                   "Treat this prediction as a guess.")
    if n_tokens >= MAX_LEN:
        st.info(f"Input was truncated to {MAX_LEN} tokens.")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=LABEL_ORDER))
    st.caption(f"Uncertainty (entropy): {s['entropy_bits']:.2f} of {s['max_entropy_bits']:.2f} bits. "
               "Probabilities have not been recalibrated and tend to be overconfident.")

with st.expander("What this model was evaluated on"):
    cond, _ = parse_condition(ckpt)
    card = CARDS.get("conditions", {}).get(cond or "", {})
    if card:
        st.write(card.get("description", ""))
        rows = card.get("results", [])
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
    for line in CARDS.get("limitations", []):
        st.write("• " + line)

st.sidebar.markdown("---")
st.sidebar.caption("Input text stays on this machine and is not stored.")
