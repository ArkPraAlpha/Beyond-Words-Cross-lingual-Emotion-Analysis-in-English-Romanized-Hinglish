# Beyond Words — App

Streamlit frontend for the Beyond Words emotion analyser. Connects to a FastAPI inference server running on Google Colab.

## What it does

- Accepts English or Romanized Hinglish text
- Sends it to the Colab inference API (Conditions A/B/C/D selectable)
- Displays emotion prediction, probability breakdown, and uncertainty flag
- Light / Dark / System theme
- Privacy notice — text is sent to Colab, not stored

## How to run

**Requirements:** Python 3.10 or 3.11. No GPU needed locally.

```powershell
# Windows PowerShell
cd app
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

```bash
# macOS / Linux
cd app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Connecting to Colab

Before analysing text, start the Colab inference server:

1. Open `../colab/Beyond_Words_Colab_Server.ipynb` in Google Colab (T4 GPU recommended)
2. Run all cells — enter your ngrok authtoken when prompted
3. Copy the **HTTPS API URL** and **API key** from the last cell output
4. In the Streamlit sidebar, click **Connect to Colab** and paste both values

## Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `bw_utils.py` | Pure-Python helpers (softmax, entropy, checkpoint discovery) |
| `model_cards.json` | Evaluation results for all 4 experimental conditions |
| `requirements.txt` | pip dependencies |

## Notes

- **Condition D** (HingRoBERTa + EmoSen) is the recommended default for Hinglish text
- Raw probabilities are **not post-hoc calibrated** — treat confidence numbers as approximate
- **No sarcasm model** is included — see the project [ROADMAP](../ROADMAP.md)
- Do not submit personal, sensitive, or private information
