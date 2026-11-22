"""Beyond Words — Emotion Analyser (Streamlit front-end).

Requires Streamlit >= 1.40  (st.dialog, st.popover, keyed containers, icons on buttons).
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from bw_utils import LABEL_ORDER, confidence_summary, load_cards

HERE = Path(__file__).resolve().parent
CARDS = load_cards(HERE / "model_cards.json")

DEFAULT_API_URL = os.environ.get("BEYOND_WORDS_API_URL", "")
DEFAULT_API_KEY = os.environ.get("BEYOND_WORDS_API_KEY", "")

EMOTION_STYLE = {
    "anger": ("#E84B5B", "😠"),
    "disgust": ("#8CA63A", "🤢"),
    "fear": ("#9857D6", "😨"),
    "joy": ("#E7A008", "😄"),
    "sadness": ("#4389E8", "😢"),
    "surprise": ("#E76A9A", "😮"),
    "neutral": ("#8993A7", "😐"),
}

PAGES = [
    ("Emotion analyser", ":material/chat_bubble_outline:"),
    ("Model info", ":material/description:"),
    ("About", ":material/info:"),
]
THEMES = [
    ("Light", ":material/light_mode:"),
    ("Dark", ":material/dark_mode:"),
    ("System", ":material/desktop_windows:"),
]
THEME_ICON = dict(THEMES)
CONDITION_ORDER = ["D", "C", "B", "A"]

EXAMPLES = [
    ("Flight delay", "I can't believe they cancelled my flight again, this is so frustrating."),
    ("Good day", "Aaj bahut accha din tha, sab dost mil gaye!"),
    ("Result worry", "Mujhe bahut dar lag raha hai kal ke result se."),
    ("Match surprise", "Yaar kal ki match dekhi? Bilkul unexpected tha!"),
]

st.set_page_config(
    page_title="Beyond Words — Emotion Analyser",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def secret_value(key: str, fallback: str) -> str:
    try:
        value = st.secrets.get(key, "")
        return str(value) if value else fallback
    except Exception:
        return fallback


def emotion_color(label: str) -> str:
    return EMOTION_STYLE.get(label.lower(), ("#8993A7", "•"))[0]


def emotion_icon(label: str) -> str:
    return EMOTION_STYLE.get(label.lower(), ("#8993A7", "•"))[1]


@st.cache_data(ttl=10, show_spinner=False)
def fetch_models(base_url: str, bearer_key: str):
    response = requests.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {bearer_key}"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def set_theme(choice: str) -> None:
    st.session_state.theme_choice = choice


def go_page(name: str) -> None:
    st.session_state.page = name


def open_connect() -> None:
    st.session_state.show_connect = True


def do_logout() -> None:
    st.session_state.connected = False
    st.session_state.api_url = ""
    st.session_state.api_key = ""
    st.session_state.pop("last_result", None)


def try_connect(url: str, key: str) -> str | None:
    """Return an error message, or None when the connection works."""
    url, key = url.strip().rstrip("/"), key.strip()
    if not url or not key:
        return "Enter both the API URL and the API key."
    try:
        requests.get(f"{url}/health", timeout=15).raise_for_status()
        models = fetch_models(url, key)
    except requests.RequestException as exc:
        return f"Connection failed. Check the URL, API key and running Colab session. Details: {exc}"
    if not models.get("models"):
        return "The API is reachable, but no model checkpoints were found."
    st.session_state.api_url = url
    st.session_state.api_key = key
    st.session_state.connected = True
    return None


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
for key, default in {
    "connected": False,
    "api_url": secret_value("BEYOND_WORDS_API_URL", DEFAULT_API_URL),
    "api_key": secret_value("BEYOND_WORDS_API_KEY", DEFAULT_API_KEY),
    "theme_choice": "Light",
    "page": "Emotion analyser",
    "show_connect": True,   # open the connect dialog on first load
    "input_text": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

theme_choice = st.session_state.theme_choice


# --------------------------------------------------------------------------- #
# Theme (Light / Dark / System) — everything in the CSS reads these variables
# --------------------------------------------------------------------------- #
LIGHT = {
    "bg": "#F4F6FC", "surface": "#FFFFFF", "surface-2": "#F7F8FD", "ink": "#171936",
    "muted": "#68718C", "line": "#DCE2F0", "input": "#FBFCFF", "chip": "#EEF1FB",
    "track": "#E9ECF4", "accent": "#3F45F0", "accent-dark": "#2F35CF", "accent-soft": "#E3E5FF",
    "ok-bg": "#DDF6E7", "ok-ink": "#17643D", "warn-bg": "#FFF0D4", "warn-ink": "#8A5B08",
    "conn-bg": "#EFFAF4", "conn-line": "#A8E7C3", "conn-ink": "#155D3B",
    "danger-bg": "#FFF0F1", "danger-line": "#F2B2B9", "danger-ink": "#D12C3F",
    "shadow": "rgba(31,43,91,.07)",
}
DARK = {
    "bg": "#0E1220", "surface": "#171D30", "surface-2": "#1B2238", "ink": "#F1F3FF",
    "muted": "#A3ACC6", "line": "#2C3652", "input": "#121829", "chip": "#222B45",
    "track": "#2A3350", "accent": "#6870FF", "accent-dark": "#535BEF", "accent-soft": "#2A2F63",
    "ok-bg": "#14392A", "ok-ink": "#7BE0A8", "warn-bg": "#3D2E0E", "warn-ink": "#F5C36B",
    "conn-bg": "#12301F", "conn-line": "#276B47", "conn-ink": "#7BE0A8",
    "danger-bg": "#3A1820", "danger-line": "#7A2E3A", "danger-ink": "#FF8A98",
    "shadow": "rgba(0,0,0,.40)",
}


def css_vars(palette: dict[str, str], scheme: str) -> str:
    body = ";".join(f"--{k}:{v}" for k, v in palette.items())
    return f":root{{{body};color-scheme:{scheme}}}"


theme_css = css_vars(LIGHT, "light")
if theme_choice == "Dark":
    theme_css += css_vars(DARK, "dark")
elif theme_choice == "System":
    theme_css += f"@media (prefers-color-scheme: dark){{{css_vars(DARK, 'dark')}}}"

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

html, body, .stApp, .stApp button, .stApp input, .stApp textarea { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--bg); color:var(--ink); transition:background .2s ease, color .2s ease; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; color:var(--ink);
}
.stApp p, .stApp label, .stApp li, .stApp summary,
.stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stMarkdownContainer"] { color:var(--ink); }
.stApp button p, .stApp button [data-testid="stMarkdownContainer"] { color:inherit; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p { color:var(--muted); }
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbarActions"],
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], .stAppDeployButton { display:none !important; }
/* keep the header/toolbar: it holds the button that re-opens the collapsed sidebar */
header[data-testid="stHeader"] { background:transparent; z-index:100; }
.stApp [data-testid="stExpandSidebarButton"], .stApp [data-testid="stSidebarCollapsedControl"] button,
.stApp [data-testid="stSidebarCollapseButton"] button {
  display:flex !important; visibility:visible !important; color:var(--ink);
  background:var(--surface); border:1px solid var(--line); border-radius:12px;
  box-shadow:0 4px 12px var(--shadow);
}
.stApp [data-testid="stExpandSidebarButton"] *, .stApp [data-testid="stSidebarCollapseButton"] button * { color:var(--ink) !important; }
.block-container { max-width:1500px; padding:1.6rem 2.2rem 3rem; }
.stApp hr { border:none; border-top:1px solid var(--line); margin:1.1rem 0; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding:.4rem .6rem 1.2rem; }
.brand { display:flex; align-items:center; gap:13px; padding:4px 4px 18px; }
.brand-mark {
  width:54px; height:54px; border-radius:17px; flex:none;
  display:flex; align-items:center; justify-content:center;
  color:#fff; font:800 20px 'Manrope',sans-serif;
  background:linear-gradient(145deg,#5964FF,#292FCB);
  box-shadow:0 8px 20px rgba(65,71,232,.25);
}
.stApp .brand-name { font:800 22px/1.05 'Manrope',sans-serif; letter-spacing:-.8px; color:var(--ink); }
.stApp .side-title { font:800 1.05rem 'Manrope',sans-serif; margin:.1rem 0 .5rem; color:var(--ink); }

.connection { border:1px solid var(--conn-line); background:var(--conn-bg); border-radius:14px; padding:13px 15px; }
.connection.off { border-color:var(--line); background:var(--surface-2); }
.stApp .connection strong { color:var(--conn-ink); font-size:1rem; }
.stApp .connection.off strong { color:var(--ink); }
.stApp .connection small { color:var(--muted); display:block; margin-top:2px; }
.dot { display:inline-block; width:13px; height:13px; border-radius:50%; background:#16C45A; margin-right:9px; vertical-align:-1px; }
.connection.off .dot { background:#9AA3B8; }
.stApp .privacy-title { font:800 1rem 'Manrope',sans-serif; margin-bottom:6px; color:var(--ink); }
.stApp .privacy { color:var(--muted); font-size:.84rem; line-height:1.5; }

/* Sidebar nav */
.stApp .st-key-nav [data-testid="stButton"] button {
  justify-content:flex-start; border:none; background:transparent; color:var(--ink);
  box-shadow:none; min-height:44px; border-radius:12px; padding:0 14px; font-weight:600;
}
.stApp .st-key-nav [data-testid="stButton"] button > div { justify-content:flex-start; gap:.7rem; }
.stApp .st-key-nav [data-testid="stButton"] button:hover { background:var(--chip); color:var(--ink); border:none; }
.stApp .st-key-nav [data-testid="stButton"] button@PRI@,
.stApp .st-key-nav [data-testid="stButton"] button@PRI@:hover {
  background:var(--accent); color:#fff; font-weight:700; box-shadow:0 6px 14px rgba(63,69,240,.25);
}
.stApp .st-key-nav [data-testid="stVerticalBlock"] { gap:.2rem; }

/* Sidebar segmented theme switch */
.stApp .st-key-theme_seg { background:var(--chip); border:1px solid var(--line); border-radius:16px; padding:4px; }
.stApp .st-key-theme_seg [data-testid="stHorizontalBlock"] { gap:.2rem; }
.stApp .st-key-theme_seg [data-testid="stButton"] button {
  border:none; background:transparent; color:var(--muted); min-height:42px;
  border-radius:12px; padding:0 4px; font-size:.82rem; font-weight:600; box-shadow:none;
}
.stApp .st-key-theme_seg [data-testid="stButton"] button:hover { color:var(--ink); background:transparent; border:none; }
.stApp .st-key-theme_seg [data-testid="stButton"] button@PRI@,
.stApp .st-key-theme_seg [data-testid="stButton"] button@PRI@:hover {
  background:var(--accent-soft); color:var(--accent); font-weight:700; box-shadow:none;
}

/* Logout */
.stApp .st-key-logout [data-testid="stButton"] button {
  background:var(--danger-bg); border:1.5px solid var(--danger-line); color:var(--danger-ink);
  border-radius:12px; min-height:48px; font-weight:800;
}
.stApp .st-key-logout [data-testid="stButton"] button:hover { border-color:var(--danger-ink); color:var(--danger-ink); }

/* ---------- Header + theme dropdown ---------- */
.stApp .eyebrow {
  color:var(--accent); text-transform:uppercase; letter-spacing:.12em;
  font-size:.76rem; font-weight:800; margin:0 0 12px;
}
.stApp .hero-title {
  font:800 clamp(2rem,3.2vw,3.25rem)/1.08 'Manrope',sans-serif;
  letter-spacing:-.055em; margin:0; color:var(--ink);
}
.stApp .hero-sub { color:var(--muted); font-size:1.12rem; margin:.65rem 0 1.4rem; }
.stApp .st-key-theme_pop { display:flex; justify-content:flex-end; }
.stApp .st-key-theme_pop button {
  border:1.5px solid var(--ink); border-radius:999px; background:var(--surface);
  color:var(--ink); min-height:44px; padding:0 18px; font-weight:600;
}
div[data-testid="stPopoverBody"] {
  min-width:200px; background:var(--surface) !important; border:1px solid var(--line) !important;
  border-radius:16px !important; padding:.5rem !important; box-shadow:0 14px 34px var(--shadow) !important;
}
.stApp .st-key-theme_menu [data-testid="stButton"] button, .st-key-theme_menu [data-testid="stButton"] button {
  justify-content:flex-start; border:none; background:transparent; color:var(--ink);
  min-height:42px; border-radius:10px; padding:0 12px; font-weight:600; width:100%;
}
.st-key-theme_menu [data-testid="stButton"] button > div { justify-content:flex-start; gap:.6rem; }
.st-key-theme_menu [data-testid="stButton"] button:hover { background:var(--chip); color:var(--ink); border:none; }
.st-key-theme_menu [data-testid="stButton"] button@PRI@,
.st-key-theme_menu [data-testid="stButton"] button@PRI@:hover { background:var(--accent-soft); color:var(--ink); font-weight:700; }

/* ---------- Input panel ---------- */
.stApp .st-key-input_panel {
  background:var(--surface); border:1px solid var(--line); border-radius:20px;
  padding:20px 24px 22px; box-shadow:0 8px 24px var(--shadow);
}
.stApp .panel-title { font:800 1.05rem 'Manrope',sans-serif; margin:0 0 10px; color:var(--ink); }
.stApp .char-count { text-align:right; color:var(--muted); font-size:.86rem; margin:6px 2px 4px; }
.stApp .try-label { color:var(--muted); font-size:.92rem; margin:0; white-space:nowrap; }
.stTextArea textarea {
  background:var(--input) !important; border:1.5px solid var(--line) !important;
  border-radius:14px !important; color:var(--ink) !important;
  font-size:1.05rem !important; padding:14px !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color:var(--muted) !important; opacity:.8; }
.stTextArea textarea:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(63,69,240,.15) !important; }
.stTextArea [data-baseweb="textarea"], .stTextArea [data-baseweb="base-input"] { background:var(--input) !important; border-radius:14px !important; border:none !important; }
.stApp .st-key-examples [data-testid="stButton"] button {
  background:var(--chip); border:1px solid var(--line); color:var(--ink);
  min-height:38px; font-size:.9rem; font-weight:500; padding:0 14px;
}
.stApp .st-key-examples [data-testid="stButton"] button:hover { border-color:var(--accent); color:var(--accent); }

/* ---------- Generic widgets ---------- */
.stApp :is([data-testid="stButton"],[data-testid="stFormSubmitButton"]) button {
  border-radius:999px; border:1px solid var(--line); background:var(--surface); color:var(--ink);
  font-weight:600; min-height:42px; transition:all .15s ease;
}
.stApp :is([data-testid="stButton"],[data-testid="stFormSubmitButton"]) button:hover { border-color:var(--accent); color:var(--accent); }
.stApp :is([data-testid="stButton"],[data-testid="stFormSubmitButton"]) button@PRI@ {
  background:var(--accent); border-color:var(--accent); color:#fff; font-weight:800;
  min-height:48px; padding:0 24px; box-shadow:0 7px 16px rgba(63,69,240,.22);
}
.stApp :is([data-testid="stButton"],[data-testid="stFormSubmitButton"]) button@PRI@:hover {
  background:var(--accent-dark); border-color:var(--accent-dark); color:#fff;
}
.stApp :is([data-testid="stButton"],[data-testid="stFormSubmitButton"]) button:disabled { opacity:.5; }

.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"] {
  background:var(--input) !important; border-color:var(--line) !important; border-radius:12px !important; color:var(--ink) !important;
}
.stApp [data-baseweb="select"] *, .stApp [data-baseweb="input"] input { color:var(--ink) !important; }
.stApp [data-baseweb="select"] svg { fill:var(--muted) !important; }
.stApp [data-baseweb="input"] input { background:transparent !important; }
.stApp [data-testid="stNumberInput"] button { color:var(--ink); }
.stApp [data-testid="stWidgetLabel"] p { font-weight:700; font-size:.92rem; }
.stApp [data-testid="stTooltipIcon"] svg { color:var(--muted); }

/* Dropdown menus & dialog are rendered in portals */
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul { background:var(--surface) !important; border:1px solid var(--line); border-radius:12px; }
[data-baseweb="popover"] li { color:var(--ink) !important; background:transparent !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background:var(--chip) !important; }
div[role="dialog"] { background:var(--surface) !important; color:var(--ink); border:1px solid var(--line); border-radius:20px !important; box-shadow:0 24px 60px rgba(0,0,0,.28); }
div[role="dialog"] h2, div[role="dialog"] button svg { color:var(--ink); }
div[data-testid="stForm"] { border:none !important; padding:0 !important; }
.stApp .dialog-sub { color:var(--muted); font-size:.92rem; margin:0 0 .6rem; }

div[data-testid="stAlert"] { border-radius:14px; background:var(--surface-2); color:var(--ink); border:1px solid var(--line); }
div[data-testid="stAlert"] * { color:var(--ink); }
[data-testid="stExpander"], [data-testid="stExpander"] details {
  border:1px solid var(--line) !important; border-radius:14px !important; background:var(--surface) !important;
}
[data-testid="stExpander"] summary p { font:800 1.05rem 'Manrope',sans-serif; }
[data-testid="stExpander"] summary svg { color:var(--ink); }

/* ---------- Result + breakdown ---------- */
.result-card { border-radius:20px; padding:26px 28px; border:1.5px solid; min-height:250px; box-shadow:0 8px 24px var(--shadow); }
.result-top { display:flex; align-items:center; gap:22px; }
.result-emoji { font-size:4.6rem; line-height:1; padding-right:22px; border-right:1.5px solid var(--line); }
.stApp .result-label { font:800 3.2rem/1 'Manrope',sans-serif; letter-spacing:-.06em; }
.stApp .result-prob { font-size:1.3rem; color:var(--muted); margin-top:8px; }
.status-chip { display:inline-block; margin-top:24px; padding:8px 15px; border-radius:999px; font-size:.95rem; font-weight:700; }
.stApp .status-ok { background:var(--ok-bg); color:var(--ok-ink); }
.stApp .status-warn { background:var(--warn-bg); color:var(--warn-ink); }
.stApp .result-note { color:var(--muted); margin:14px 0 0; line-height:1.5; font-size:1.02rem; }
.breakdown { background:var(--surface); border:1px solid var(--line); border-radius:20px; padding:24px 28px; box-shadow:0 8px 24px var(--shadow); min-height:250px; }
.stApp .breakdown h4 { font:800 1.3rem 'Manrope',sans-serif; margin:0 0 14px; }
.prob-row { display:grid; grid-template-columns:105px 1fr 52px; align-items:center; gap:14px; margin:13px 0; }
.stApp .prob-name { font-weight:800; }
.prob-track { height:13px; border-radius:99px; background:var(--track); overflow:hidden; }
.prob-fill { height:100%; border-radius:99px; }
.stApp .prob-pct { text-align:right; color:var(--muted); font-variant-numeric:tabular-nums; font-weight:500; }
.stApp .meta-line { color:var(--muted); font-size:.92rem; margin:18px 0 14px; }

/* ---------- Info pages ---------- */
.stApp .panel { background:var(--surface); border:1px solid var(--line); border-radius:20px; padding:22px 26px; box-shadow:0 8px 24px var(--shadow); margin-bottom:1rem; }
.stApp .panel h4 { font:800 1.2rem 'Manrope',sans-serif; margin:0 0 .6rem; }
.stApp .panel p, .stApp .panel li { color:var(--ink); line-height:1.6; }
.bw-table { width:100%; border-collapse:collapse; font-size:.92rem; margin-top:.5rem; }
.stApp .bw-table th { text-align:left; color:var(--muted); font-weight:700; padding:8px 10px; border-bottom:1px solid var(--line); }
.stApp .bw-table td { padding:8px 10px; border-bottom:1px solid var(--line); color:var(--ink); }

@media(max-width:800px) {
  .block-container { padding:1rem 1rem 2rem; }
  .result-label { font-size:2.3rem !important; }
  .result-emoji { font-size:3.2rem; padding-right:14px; }
  .prob-row { grid-template-columns:78px 1fr 44px; gap:8px; }
}
@media(prefers-reduced-motion:reduce) { * { transition:none !important; } }
"""
PRIMARY = ':is([kind="primary"],[data-testid^="stBaseButton-primary"])'
st.markdown(f"<style>{BASE_CSS.replace('@PRI@', PRIMARY)}{theme_css}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Connect dialog
# --------------------------------------------------------------------------- #
@st.dialog("🔑 Connect to Colab API")
def connect_dialog() -> None:
    st.markdown('<p class="dialog-sub">Enter your Colab inference endpoint details to continue.</p>', unsafe_allow_html=True)
    with st.form("connect_form"):
        url = st.text_input("API URL", value=st.session_state.api_url,
                            placeholder="https://your-colab-endpoint.ngrok-free.app")
        key = st.text_input("API Key", value=st.session_state.api_key,
                            type="password", placeholder="Enter your API key")
        _, c_cancel, c_connect = st.columns([2, 1, 1])
        cancelled = c_cancel.form_submit_button("Cancel", use_container_width=True)
        submitted = c_connect.form_submit_button("Connect", type="primary", use_container_width=True)
    if cancelled:
        st.rerun()
    if submitted:
        error = try_connect(url, key)
        if error:
            st.error(error)
        else:
            st.rerun()


# --------------------------------------------------------------------------- #
# Server state (fetched before the sidebar so model settings can live there)
# --------------------------------------------------------------------------- #
health = None
model_data: dict = {}
conn_error: Exception | None = None
if st.session_state.connected:
    try:
        resp = requests.get(f"{st.session_state.api_url}/health", timeout=15)
        resp.raise_for_status()
        health = resp.json()
        model_data = fetch_models(st.session_state.api_url, st.session_state.api_key)
    except requests.RequestException as exc:
        conn_error = exc

available = model_data.get("models", []) if model_data else []
condition_map = {item["condition"]: item for item in available}
conditions = [c for c in CONDITION_ORDER if c in condition_map]
conditions += [c for c in condition_map if c not in conditions]
condition = None
seed = None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">BW</div>'
        '<div class="brand-name">Beyond<br>Words</div></div>',
        unsafe_allow_html=True,
    )

    with st.container(key="nav"):
        for name, icon in PAGES:
            st.button(
                name, key=f"nav_{name}", icon=icon, use_container_width=True,
                type="primary" if st.session_state.page == name else "secondary",
                on_click=go_page, args=(name,),
            )

    st.markdown("---")

    if st.session_state.connected and not conn_error:
        st.markdown(
            '<div class="connection"><strong><span class="dot"></span>Colab connected</strong>'
            '<small>Google Colab inference API</small></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="connection off"><strong><span class="dot"></span>Not connected</strong>'
            '<small>Connect to your Colab inference API</small></div>',
            unsafe_allow_html=True,
        )

    if conditions:
        st.markdown("---")
        st.markdown('<div class="side-title">Model settings</div>', unsafe_allow_html=True)
        condition = st.selectbox(
            "Experimental condition",
            options=conditions,
            index=conditions.index("D") if "D" in conditions else 0,
            format_func=lambda c: (
                f"Condition {c} — {condition_map[c]['name']}" + (" (Recommended)" if c == "D" else "")
            ),
            help="Which training setup to use. Condition D is recommended for Hinglish-focused demos.",
            key="condition",
        )
        seeds = sorted(int(x) for x in condition_map[condition].get("seeds", []))
        if seeds:
            seed = st.selectbox(
                "Training seed",
                options=seeds,
                index=seeds.index(42) if 42 in seeds else 0,
                help="Random seed used when this checkpoint was trained.",
                key=f"seed_{condition}",
            )
        else:
            st.error(f"No seeds are registered for condition {condition}.")
        if health:
            st.caption(f"Connected device: {health.get('device', 'Colab')}")

    st.markdown("---")
    st.markdown('<div class="side-title">Appearance</div>', unsafe_allow_html=True)
    with st.container(key="theme_seg"):
        seg_cols = st.columns(3)
        for col, (name, icon) in zip(seg_cols, THEMES):
            with col:
                st.button(
                    name, key=f"seg_{name}", icon=icon, use_container_width=True,
                    type="primary" if theme_choice == name else "secondary",
                    on_click=set_theme, args=(name,),
                )

    st.markdown("---")
    st.markdown(
        '<div class="privacy-title">🔒 Privacy reminder</div>'
        '<div class="privacy">Don\'t enter sensitive, personal, or confidential information. '
        "Your text is sent to a Google Colab inference API for emotion classification.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    if st.session_state.connected:
        with st.container(key="logout"):
            st.button("Logout", icon=":material/logout:", use_container_width=True, on_click=do_logout)
    else:
        st.button("Connect to Colab", icon=":material/key:", type="primary",
                  use_container_width=True, on_click=open_connect)


# --------------------------------------------------------------------------- #
# Page header with theme dropdown (top-right)
# --------------------------------------------------------------------------- #
def page_header(eyebrow: str, title: str, sub: str) -> None:
    left, right = st.columns([5, 1.2], vertical_alignment="top")
    left.markdown(
        f'<div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="hero-title">{html.escape(title)}</h1>'
        f'<p class="hero-sub">{html.escape(sub)}</p>',
        unsafe_allow_html=True,
    )
    with right:
        with st.container(key="theme_pop"):
            with st.popover(theme_choice, icon=THEME_ICON[theme_choice]):
                with st.container(key="theme_menu"):
                    for name, icon in THEMES:
                        label = f"{name}\u2003\u2003✓" if name == theme_choice else name
                        st.button(
                            label, key=f"menu_{name}", icon=icon, use_container_width=True,
                            type="primary" if name == theme_choice else "secondary",
                            on_click=set_theme, args=(name,),
                        )


def results_table(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_html(index=False, border=0, classes="bw-table", escape=True)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def render_result(entry: dict) -> None:
    result = entry["result"]
    probabilities = result.get("probabilities", [])
    labels = result.get("labels", LABEL_ORDER)
    summary = confidence_summary(probabilities, labels=labels)
    top_label = summary["label"]
    color = emotion_color(top_label)
    uncertain = summary["flag"] == "uncertain"

    if result.get("truncated", False):
        st.warning(
            f"The input exceeded the model's {result.get('max_length', 128)}-token "
            "limit and was truncated."
        )

    ranked = sorted(zip(labels, probabilities), key=lambda pair: pair[1], reverse=True)
    result_col, breakdown_col = st.columns([0.9, 1.1], gap="large")

    with result_col:
        chip_class = "status-warn" if uncertain else "status-ok"
        chip_text = "Uncertain result" if uncertain else "✓ Confidence checks passed"
        if uncertain:
            note = (
                f"The top two emotions are close. Runner-up: "
                f"{html.escape(summary['runner_up'].title())} "
                f"({summary['runner_up_prob']:.1%}). Treat this result as uncertain."
            )
        else:
            note = "This is a heuristic estimate, not a guarantee of correctness."
        st.markdown(
            f'<div class="result-card" style="border-color:{color};'
            f'background:linear-gradient(135deg,{color}24,var(--surface) 80%);">'
            f'<div class="result-top"><div class="result-emoji">{emotion_icon(top_label)}</div>'
            f'<div><div class="result-label" style="color:var(--ink)">{html.escape(top_label.title())}</div>'
            f'<div class="result-prob">{summary["prob"]:.0%} probability</div></div></div>'
            f'<span class="status-chip {chip_class}">{chip_text}</span>'
            f'<p class="result-note">{note}</p></div>',
            unsafe_allow_html=True,
        )

    with breakdown_col:
        rows = "".join(
            f'<div class="prob-row"><div class="prob-name">{html.escape(label.title())}</div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{max(prob * 100, 2.5):.1f}%;'
            f'background:{emotion_color(label)}"></div></div>'
            f'<div class="prob-pct">{prob * 100:.0f}%</div></div>'
            for label, prob in ranked
        )
        st.markdown(f'<div class="breakdown"><h4>Emotion breakdown</h4>{rows}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="meta-line">ⓘ &nbsp; Condition {entry["condition"]} &nbsp; • &nbsp; '
        f'Seed {entry["seed"]} &nbsp; • &nbsp; {html.escape(str(result.get("device", "Colab")))} inference</div>',
        unsafe_allow_html=True,
    )


def set_example(text: str) -> None:
    st.session_state.input_text = text


def analyser_page() -> None:
    page_header(
        "Emotion intelligence / research preview",
        "Understand the emotion behind the words.",
        "Analyse English and Romanized Hinglish in seconds.",
    )

    if not st.session_state.connected:
        st.info("Connect your running Colab session to start analysing text.")
        return
    if conn_error:
        st.error("Lost connection to the Colab inference server. Use Logout and reconnect with the current tunnel URL and API key.")
        with st.expander("Connection details"):
            st.code(str(conn_error))
        return
    if not conditions:
        st.error("No model checkpoints were found by the Colab server.")
        return

    with st.container(key="input_panel"):
        st.markdown('<div class="panel-title">Your text</div>', unsafe_allow_html=True)
        text = st.text_area(
            "Text to analyse",
            key="input_text",
            height=110,
            max_chars=5000,
            placeholder="Type a sentence in English or Romanized Hinglish...",
            label_visibility="collapsed",
        )
        st.markdown(f'<div class="char-count">{len(text)} / 5,000</div>', unsafe_allow_html=True)

        with st.container(key="examples"):
            cols = st.columns([1.15, 1, 1, 1, 1.15, 0.6, 1.6], vertical_alignment="center")
            cols[0].markdown('<p class="try-label">Try an example:</p>', unsafe_allow_html=True)
            for col, (name, example) in zip(cols[1:5], EXAMPLES):
                col.button(name, key=f"ex_{name}", use_container_width=True,
                           on_click=set_example, args=(example,))
            analyse = cols[6].button(
                "Analyse emotion  →", type="primary", use_container_width=True,
                disabled=not bool(text.strip()), key="analyse",
            )

    if analyse and seed is not None:
        payload = {"condition": condition, "seed": int(seed), "text": text.strip(), "temperature": 1.0}
        try:
            with st.spinner(f"Running Condition {condition}, seed {seed}, on Google Colab..."):
                response = requests.post(
                    f"{st.session_state.api_url}/predict",
                    json=payload,
                    headers={"Authorization": f"Bearer {st.session_state.api_key}"},
                    timeout=120,
                )
                response.raise_for_status()
                st.session_state.last_result = {
                    "result": response.json(), "condition": condition, "seed": int(seed),
                }
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except Exception:
                pass
            st.error(f"Prediction request failed: {detail or exc}")
        except requests.RequestException as exc:
            st.error(f"Could not complete prediction: {exc}")

    # Result is kept in session state so it survives theme changes and other reruns.
    if "last_result" in st.session_state:
        render_result(st.session_state.last_result)

    with st.expander("Limitations & responsible use"):
        st.write(
            "This is an emotion-classification research prototype, not a detector of a "
            "person's true internal feelings."
        )
        for limitation in CARDS.get("limitations", []):
            st.write(f"• {limitation}")
        st.write("No sarcasm, sentiment, or sentiment-intent model is included in this prototype.")

    st.caption(
        "Privacy: text entered here is sent to the configured Colab inference endpoint for prediction. "
        "Do not enter names, contact details, private conversations, or other sensitive information. "
        "The app does not intentionally save submitted text."
    )


def model_info_page() -> None:
    page_header("Emotion intelligence / research preview", "Model info",
                "Experimental conditions and their evaluation results.")
    card_conditions = CARDS.get("conditions", {})
    ordered = [c for c in CONDITION_ORDER if c in card_conditions]
    ordered += [c for c in card_conditions if c not in ordered]
    if not ordered:
        st.info("No model cards were found.")
    for c in ordered:
        card = card_conditions[c]
        title = f"Condition {c} — {card.get('name', c)}" + (" (Recommended)" if c == "D" else "")
        with st.expander(title, expanded=(c == (condition or "D"))):
            if c == "D":
                st.caption(
                    "Recommended default for a Hinglish-focused demonstration. Its advantage over "
                    "Condition C is not conclusive on the small HinEmo-9 test set."
                )
            st.write(card.get("description", ""))
            if card.get("results"):
                st.markdown(results_table(card["results"]), unsafe_allow_html=True)


def about_page() -> None:
    page_header("Emotion intelligence / research preview", "About Beyond Words",
                "A research prototype for emotion classification in English and Romanized Hinglish.")
    st.markdown(
        '<div class="panel"><h4>What it does</h4>'
        "<p>Beyond Words classifies short texts into seven emotions — joy, surprise, neutral, sadness, "
        "anger, fear and disgust — and shows the probability the model assigns to each one.</p></div>"
        '<div class="panel"><h4>How it works</h4>'
        "<p>This app is a thin front-end. Your text is sent to a model server running in Google Colab, "
        "which returns class probabilities. A confidence check flags results where the top two emotions are close.</p></div>"
        '<div class="panel"><h4>Use responsibly</h4>'
        "<p>Results are heuristic estimates, not a guarantee of correctness or a reading of anyone's true feelings. "
        "Don't submit sensitive or personal information.</p></div>",
        unsafe_allow_html=True,
    )


# Open the connect dialog once when not connected (flag is cleared first so
# closing the dialog with the X does not make it pop up again).
if not st.session_state.connected and st.session_state.show_connect:
    st.session_state.show_connect = False
    connect_dialog()

page = st.session_state.page
if page == "Model info":
    model_info_page()
elif page == "About":
    about_page()
else:
    analyser_page()
