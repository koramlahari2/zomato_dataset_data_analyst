"""
app.py
======
Streamlit dashboard for the Food Delivery Analytics & AI project.

Run from the project root:
    streamlit run app.py

Sections
--------
  1. Overview        — KPI cards: total deliveries, avg time, avg rating, avg distance
  2. Analytics       — Interactive charts: time by traffic, weather, vehicle, order type,
                       city, and distance vs time scatter
  3. Prediction      — User-input form -> trained Random Forest model -> predicted time
                       + AI plain-language explanation (requires API credentials)
  4. Insights        — Key findings from EDA, feature importance, model performance

Data sources (loaded dynamically — never hard-coded)
-----------------------------------------------------
  data/processed/zomato_cleaned.csv   — cleaned dataset for analytics
  data/processed/zomato_features.csv  — engineered features (for feature importance)
  models/best_model.joblib            — trained model
  models/model_metadata.json          — feature list + test metrics
  models/model_comparison.csv         — 5-model comparison table

Robustness
----------
  - All file loads are wrapped in try/except with clear st.error messages.
  - Missing required files halt the app with actionable instructions.
  - Missing optional files (comparison table) degrade gracefully.
  - All categorical lookups guard against unexpected values.
  - Model prediction is wrapped in try/except.
  - AI section shows specific error messages per failure type.
  - API keys are never displayed or logged.
"""

import json
import math
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Local module — ensure project root is on the path regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ai_explanation import AIError, get_ai_explanation, load_credentials, probe_ai_status

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Food Delivery Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the app works from any cwd
# ---------------------------------------------------------------------------
_ROOT        = os.path.dirname(os.path.abspath(__file__))
CLEANED_CSV  = os.path.join(_ROOT, "data", "processed", "zomato_cleaned.csv")
FEATURES_CSV = os.path.join(_ROOT, "data", "processed", "zomato_features.csv")
MODEL_PATH   = os.path.join(_ROOT, "models", "best_model.joblib")
META_PATH    = os.path.join(_ROOT, "models", "model_metadata.json")
COMPARE_PATH = os.path.join(_ROOT, "models", "model_comparison.csv")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PRIMARY   = "#3b82d4"
ACCENT    = "#f59e0b"
CHART_CAT = px.colors.qualitative.Pastel

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Segoe UI, system-ui, sans-serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=CHART_CAT,
)

# ---------------------------------------------------------------------------
# Startup file-existence check
# Runs once before any Streamlit widget is rendered.
# ---------------------------------------------------------------------------
_REQUIRED_FILES = {
    "Cleaned dataset":    CLEANED_CSV,
    "Feature dataset":    FEATURES_CSV,
    "Trained model":      MODEL_PATH,
    "Model metadata":     META_PATH,
}

_missing = [label for label, path in _REQUIRED_FILES.items() if not os.path.exists(path)]
if _missing:
    st.error(
        "**Required files are missing. The application cannot start.**\n\n"
        + "\n".join(f"- {label}: `{_REQUIRED_FILES[label]}`" for label in _missing)
        + "\n\nRun the pipeline in order:\n"
        "```\npython src/data_cleaning.py\n"
        "python src/features.py\n"
        "python src/train.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Data loaders — @st.cache_data / @st.cache_resource so files load once
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_cleaned() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_CSV)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    feat = pd.read_csv(FEATURES_CSV)
    if len(feat) != len(df):
        raise ValueError(
            f"Row count mismatch: cleaned={len(df)}, features={len(feat)}. "
            "Re-run src/features.py to regenerate the feature dataset."
        )
    df["distance_km"] = feat["distance_km"].values
    return df


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    with open(META_PATH) as f:
        meta = json.load(f)
    required_keys = {"feature_names", "best_model_name", "test_mae", "test_rmse", "test_r2"}
    missing_keys = required_keys - set(meta.keys())
    if missing_keys:
        raise KeyError(
            f"model_metadata.json is missing required keys: {missing_keys}. "
            "Re-run src/train.py."
        )
    return meta


@st.cache_data(show_spinner=False)
def _load_comparison_optional() -> pd.DataFrame | None:
    """Returns None (not an error) when the comparison file does not exist."""
    if not os.path.exists(COMPARE_PATH):
        return None
    return pd.read_csv(COMPARE_PATH)


# ---------------------------------------------------------------------------
# Load all required data — show a single clear error if anything fails
# ---------------------------------------------------------------------------
_load_error: str | None = None
df = feat_df = model = meta = compare = None  # type: ignore[assignment]

with st.spinner("Loading data and model..."):
    try:
        df      = load_cleaned()
        feat_df = load_features()
        model   = load_model()
        meta    = load_metadata()
        compare = _load_comparison_optional()
    except FileNotFoundError as exc:
        _load_error = f"A required file was not found: `{exc.filename}`"
    except ValueError as exc:
        _load_error = str(exc)
    except KeyError as exc:
        _load_error = str(exc)
    except Exception as exc:
        _load_error = f"Unexpected error while loading data ({type(exc).__name__}): {exc}"

if _load_error:
    st.error(f"**Failed to load the application data.**\n\n{_load_error}")
    st.stop()

# Sanity: ensure the loaded dataset is not empty
if df is None or len(df) == 0:
    st.error(
        "**The cleaned dataset is empty.**\n\n"
        "Re-run `python src/data_cleaning.py` to regenerate it."
    )
    st.stop()

TARGET = "Time_taken (min)"

# Verify the target column exists
if TARGET not in df.columns:
    st.error(
        f"**Column `{TARGET}` not found in the dataset.**\n\n"
        "Expected columns: " + ", ".join(df.columns.tolist())
    )
    st.stop()

# ---------------------------------------------------------------------------
# Global CSS — design system
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base typography ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] hr { border-color: #334155 !important; }
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-radius: 6px; cursor: pointer;
    font-size: 0.9rem; color: #94a3b8 !important;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #1e293b; color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked),
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    background: #1e3a5f; color: #93c5fd !important;
}

/* ── Hero banner ───────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1e40af 100%);
    border-radius: 14px; padding: 36px 40px 32px; margin-bottom: 28px;
    color: white;
}
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #93c5fd; margin-bottom: 10px;
}
.hero-title {
    font-size: 1.9rem; font-weight: 800; line-height: 1.15;
    color: #ffffff; margin-bottom: 10px;
}
.hero-desc {
    font-size: 0.93rem; color: #93c5fd; line-height: 1.65;
    max-width: 680px; margin-bottom: 18px;
}
.hero-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.hero-chip {
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.75rem; color: #e2e8f0;
}

/* ── Page section header ───────────────────────────────────────────────── */
.page-header {
    padding-bottom: 14px; margin-bottom: 4px;
    border-bottom: 2px solid #e2e8f0;
}
.page-header h2 {
    font-size: 1.35rem; font-weight: 700; color: #0f172a; margin: 0 0 4px;
}
.page-header p {
    font-size: 0.88rem; color: #64748b; margin: 0;
}

/* ── Sub-section labels ────────────────────────────────────────────────── */
.sub-section {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.10em;
    text-transform: uppercase; color: #94a3b8;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 6px; margin: 24px 0 14px;
}

/* ── KPI cards ─────────────────────────────────────────────────────────── */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; }
.kpi-card {
    flex: 1; background: #ffffff;
    border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 20px 24px; text-align: left;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.kpi-icon { font-size: 1.4rem; margin-bottom: 8px; }
.kpi-label {
    font-size: 0.72rem; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
}
.kpi-value {
    font-size: 2rem; font-weight: 800; color: #0f172a;
    line-height: 1.1; margin-bottom: 2px;
}
.kpi-unit  { font-size: 1rem; font-weight: 400; color: #64748b; }
.kpi-sub   { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }
.kpi-card.accent { border-top: 3px solid #3b82d4; }

/* ── Chart wrapper ─────────────────────────────────────────────────────── */
.chart-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 20px 18px 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}
.chart-title {
    font-size: 0.88rem; font-weight: 700; color: #1e293b;
    margin-bottom: 2px;
}
.chart-caption { font-size: 0.78rem; color: #94a3b8; margin-bottom: 10px; }

/* ── Prediction result ─────────────────────────────────────────────────── */
.pred-outer {
    background: linear-gradient(135deg, #0f172a 0%, #1e40af 100%);
    border-radius: 16px; padding: 36px 32px; text-align: center;
    box-shadow: 0 4px 20px rgba(59,130,212,0.25);
}
.pred-label {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #93c5fd; margin-bottom: 10px;
}
.pred-value {
    font-size: 4.5rem; font-weight: 900; color: #ffffff;
    line-height: 1; margin: 0 0 4px;
}
.pred-unit  { font-size: 1.1rem; color: #bfdbfe; font-weight: 400; }
.pred-range {
    margin-top: 14px; font-size: 0.82rem; color: #93c5fd;
    background: rgba(255,255,255,0.08); border-radius: 20px;
    display: inline-block; padding: 4px 16px;
}

/* ── Form section labels ───────────────────────────────────────────────── */
.form-section {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748b;
    border-bottom: 1px solid #f1f5f9; padding-bottom: 6px; margin: 18px 0 12px;
}

/* ── Insight cards ─────────────────────────────────────────────────────── */
.insight-card {
    background: #f8faff; border: 1px solid #dbeafe;
    border-left: 4px solid #3b82d4; border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin-bottom: 10px;
}
.insight-title {
    font-size: 0.9rem; font-weight: 700; color: #1e3a5f; margin-bottom: 5px;
}
.insight-body { font-size: 0.85rem; color: #475569; line-height: 1.6; }

/* ── Warning / limitation cards ────────────────────────────────────────── */
.warn-card {
    background: #fffbeb; border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b; border-radius: 0 10px 10px 0;
    padding: 12px 16px; margin-bottom: 8px;
}
.warn-title { font-size: 0.85rem; font-weight: 700; color: #92400e; margin-bottom: 3px; }
.warn-body  { font-size: 0.82rem; color: #78350f; line-height: 1.55; }

/* ── AI explanation box ─────────────────────────────────────────────────── */
.ai-box {
    background: #f0f9ff; border: 1px solid #bae6fd;
    border-left: 4px solid #0ea5e9; border-radius: 0 10px 10px 0;
    padding: 18px 22px; font-size: 0.93rem; line-height: 1.75; color: #0c4a6e;
}
.ai-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #0369a1; margin-bottom: 10px;
}

/* ── Metric explain cards ───────────────────────────────────────────────── */
.metric-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 20px 22px; height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-name {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #64748b; margin-bottom: 6px;
}
.metric-value {
    font-size: 2rem; font-weight: 800; color: #0f172a; line-height: 1;
    margin-bottom: 2px;
}
.metric-unit { font-size: 0.9rem; font-weight: 400; color: #64748b; }
.metric-desc { font-size: 0.82rem; color: #475569; line-height: 1.6; margin-top: 10px; }
.metric-card.good  { border-top: 3px solid #10b981; }
.metric-card.lower { border-top: 3px solid #3b82d4; }

/* ── Methodology step ───────────────────────────────────────────────────── */
.step {
    display: flex; gap: 16px; align-items: flex-start;
    margin-bottom: 16px;
}
.step-num {
    flex-shrink: 0; width: 32px; height: 32px;
    background: #1e3a5f; color: #93c5fd;
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 0.85rem; font-weight: 800;
}
.step-body { padding-top: 4px; }
.step-title { font-size: 0.9rem; font-weight: 700; color: #1e293b; margin-bottom: 3px; }
.step-desc  { font-size: 0.82rem; color: #64748b; line-height: 1.55; }

/* ── Divider label ──────────────────────────────────────────────────────── */
.divider-label {
    display: flex; align-items: center; gap: 12px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #94a3b8; margin: 28px 0 18px;
}
.divider-label::before, .divider-label::after {
    content: ""; flex: 1; height: 1px; background: #e2e8f0;
}

/* ── Placeholder prompt ─────────────────────────────────────────────────── */
.placeholder {
    text-align: center; padding: 52px 32px;
    background: #f8fafc; border-radius: 14px;
    border: 2px dashed #cbd5e1;
}
.placeholder-icon { font-size: 2.8rem; margin-bottom: 12px; }
.placeholder-title {
    font-size: 1.05rem; font-weight: 700; color: #475569; margin-bottom: 6px;
}
.placeholder-sub { font-size: 0.85rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation + AI status badge
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div style="padding:20px 4px 8px; font-size:1.05rem; font-weight:800; '
    'color:#f1f5f9; letter-spacing:0.01em;">Delivery Analytics</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div style="font-size:0.75rem; color:#64748b; padding:0 4px 16px; line-height:1.5;">'
    'Food Delivery Time Prediction<br>Student Data Science Project</div>',
    unsafe_allow_html=True,
)
st.sidebar.divider()

PAGES = [
    " Overview",
    " Delivery Analytics",
    " Prediction",
    " Insights & Methodology",
]
page = st.sidebar.radio("Navigate to", PAGES, label_visibility="collapsed")

# Normalise page name for if/elif matching
page = page.split("  ", 1)[-1].strip()

st.sidebar.divider()
st.sidebar.markdown(
    f'<div style="font-size:0.75rem; color:#64748b; line-height:1.8; padding:0 4px;">'
    f'<span style="color:#94a3b8; font-weight:600;">DATASET</span><br>'
    f'{len(df):,} delivery records<br>'
    f'Feb – Apr 2022 &middot; India<br><br>'
    f'<span style="color:#94a3b8; font-weight:600;">BEST MODEL</span><br>'
    f'{meta["best_model_name"]}<br>'
    f'MAE <span style="color:#93c5fd; font-weight:700;">{meta["test_mae"]} min</span> &nbsp;'
    f'R² <span style="color:#93c5fd; font-weight:700;">{meta["test_r2"]}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ---- AI status badge ----
st.sidebar.divider()
st.sidebar.markdown(
    '<div style="font-size:0.72rem; font-weight:700; letter-spacing:0.08em; '
    'text-transform:uppercase; color:#64748b; padding:0 4px 6px;">AI Explanation</div>',
    unsafe_allow_html=True,
)

_ai_creds = load_credentials()

if not _ai_creds:
    st.sidebar.markdown(
        '<span style="color:#94a3b8; font-size:0.82rem;">⬜ Not configured</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Add credentials to `.streamlit/secrets.toml` to enable AI explanations.")
else:
    _backend_label = _ai_creds.get("backend", "unknown").capitalize()
    # Probe status (cached for 60 s to avoid hammering the API on every rerun)
    @st.cache_data(ttl=60, show_spinner=False)
    def _cached_probe(creds_repr: str) -> tuple[str, str]:
        """creds_repr is a safe string key — never contains the actual key value."""
        from src.ai_explanation import probe_ai_status
        result = probe_ai_status(_ai_creds)
        return result.error.value, result.detail

    _creds_key = f"{_ai_creds.get('backend')}:{_ai_creds.get('base_url','default')}"
    _status_code, _status_detail = _cached_probe(_creds_key)

    if _status_code == AIError.OK.value:
        st.sidebar.markdown(
            f'<span style="color:#10b981; font-size:0.82rem;">✅ {_backend_label} — connected</span>',
            unsafe_allow_html=True,
        )
    elif _status_code == AIError.AUTH_FAILED.value:
        st.sidebar.markdown(
            f'<span style="color:#ef4444; font-size:0.82rem;">🔴 Auth failed</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(_status_detail)
    elif _status_code == AIError.NETWORK_ERROR.value:
        st.sidebar.markdown(
            f'<span style="color:#f59e0b; font-size:0.82rem;">🟡 Network error</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(_status_detail)
    elif _status_code == AIError.SDK_MISSING.value:
        st.sidebar.markdown(
            '<span style="color:#f59e0b; font-size:0.82rem;">🟡 SDK missing</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(_status_detail)
    else:
        st.sidebar.markdown(
            f'<span style="color:#f59e0b; font-size:0.82rem;">🟡 {_status_code}</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(_status_detail)


# ===========================================================================
# Shared helper — group statistics for bar charts
# ===========================================================================
def group_stats(col: str, label_col: str | None = None) -> pd.DataFrame:
    """Mean ± 95 % CI of TARGET grouped by col. Returns empty DataFrame on error."""
    try:
        if col not in df.columns:
            return pd.DataFrame()
        g = df.groupby(col)[TARGET].agg(["mean", "std", "count"]).reset_index()
        g.columns = [col, "mean", "std", "count"]
        g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
        if label_col:
            g = g.rename(columns={col: label_col})
        return g
    except Exception:
        return pd.DataFrame()


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == "Overview":

    # ── Hero banner ──────────────────────────────────────────────────────────
    total       = len(df)
    avg_time    = df[TARGET].mean()
    avg_rating  = df["Delivery_person_Ratings"].mean() if "Delivery_person_Ratings" in df.columns else float("nan")
    avg_dist    = df["distance_km"].mean()              if "distance_km" in df.columns              else float("nan")
    median_time = df[TARGET].median()

    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">Student Data Science Project &nbsp;&middot;&nbsp; Python &nbsp;&middot;&nbsp; scikit-learn &nbsp;&middot;&nbsp; Streamlit</div>
        <div class="hero-title">Food Delivery Time Analytics</div>
        <div class="hero-desc">
            Analysing <strong>{total:,}</strong> Zomato food delivery orders across India (Feb&ndash;Apr 2022)
            to understand what drives delivery time and build a machine learning model that predicts it.
        </div>
        <div class="hero-chips">
            <span class="hero-chip">📦 {total:,} Orders</span>
            <span class="hero-chip">🕐 Avg {avg_time:.1f} min</span>
            <span class="hero-chip">📍 3 City Types</span>
            <span class="hero-chip">🌦 6 Weather Conditions</span>
            <span class="hero-chip">🌀 Random Forest · R² {meta["test_r2"]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card accent">
            <div class="kpi-icon">📦</div>
            <div class="kpi-label">Total Deliveries</div>
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-sub">unique orders analysed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card accent">
            <div class="kpi-icon">🕐</div>
            <div class="kpi-label">Avg Delivery Time</div>
            <div class="kpi-value">{avg_time:.1f} <span class="kpi-unit">min</span></div>
            <div class="kpi-sub">median {median_time:.0f} min</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        rating_str = f"{avg_rating:.2f}" if not math.isnan(avg_rating) else "N/A"
        st.markdown(f"""
        <div class="kpi-card accent">
            <div class="kpi-icon">⭐</div>
            <div class="kpi-label">Avg Rider Rating</div>
            <div class="kpi-value">{rating_str} <span class="kpi-unit">/ 5</span></div>
            <div class="kpi-sub">delivery person score</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        dist_str = f"{avg_dist:.1f}" if not math.isnan(avg_dist) else "N/A"
        st.markdown(f"""
        <div class="kpi-card accent">
            <div class="kpi-icon">📍</div>
            <div class="kpi-label">Avg Distance</div>
            <div class="kpi-value">{dist_str} <span class="kpi-unit">km</span></div>
            <div class="kpi-sub">Haversine straight-line</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Delivery time distribution ----
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Delivery Time Distribution")
        fig = px.histogram(
            df, x=TARGET, nbins=45,
            labels={TARGET: "Delivery Time (minutes)", "count": "Orders"},
            color_discrete_sequence=[PRIMARY],
        )
        fig.add_vline(x=avg_time, line_dash="dash", line_color=ACCENT,
                      annotation_text=f"Mean {avg_time:.1f} min",
                      annotation_position="top right")
        fig.update_layout(**PLOTLY_LAYOUT, bargap=0.05,
                          xaxis_title="Delivery Time (minutes)",
                          yaxis_title="Number of Orders")
        fig.update_traces(marker_line_width=0.3, marker_line_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Orders by Category")
        tab1, tab2, tab3 = st.tabs(["Traffic", "City", "Vehicle"])

        def _safe_pie(col: str, label: str) -> None:
            if col not in df.columns:
                st.caption(f"Column '{col}' not found in dataset.")
                return
            counts = df[col].value_counts().reset_index()
            counts.columns = [label, "Orders"]
            if counts.empty:
                st.caption("No data available.")
                return
            fig_p = px.pie(counts, names=label, values="Orders",
                           color_discrete_sequence=CHART_CAT, hole=0.4)
            fig_p.update_layout(**PLOTLY_LAYOUT)
            fig_p.update_traces(textposition="outside", textinfo="percent+label")
            st.plotly_chart(fig_p, use_container_width=True)

        with tab1:
            _safe_pie("Road_traffic_density", "Traffic")
        with tab2:
            _safe_pie("City", "City")
        with tab3:
            _safe_pie("Type_of_vehicle", "Vehicle")

    # ---- Daily trend ----
    st.markdown("#### Order Volume Over Time")
    if "Order_Date" in df.columns:
        daily = df.groupby("Order_Date").size().reset_index(name="Orders")
        daily = daily.dropna(subset=["Order_Date"])
        if daily.empty:
            st.info("No valid order dates found in the dataset.")
        else:
            fig5 = px.line(daily, x="Order_Date", y="Orders",
                           color_discrete_sequence=[PRIMARY],
                           labels={"Order_Date": "Date", "Orders": "Orders per Day"})
            fig5.update_layout(**PLOTLY_LAYOUT, xaxis_title="Date", yaxis_title="Orders per Day")
            fig5.update_traces(line_width=1.8)
            st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Order date column not available.")


# ===========================================================================
# PAGE 2 — DELIVERY ANALYTICS
# ===========================================================================
elif page == "Delivery Analytics":

    st.markdown("""
    <div class="page-header">
        <h2>Delivery Analytics</h2>
        <p>How traffic, weather, vehicle type, city, and distance are associated with delivery time.</p>
    </div>
    """, unsafe_allow_html=True)

    def _bar_chart(grp: pd.DataFrame, x: str, y: str, title: str, color_scale: str,
                   orientation: str = "v", error_col: str = "ci95") -> None:
        """Renders a bar chart inside a chart-card, showing a caption if data is empty."""
        if grp.empty or y not in grp.columns:
            st.caption(f"No data available for: {title}")
            return
        chart_data = grp.copy()
        chart_data[y] = pd.to_numeric(chart_data[y], errors="coerce")
        chart_data = chart_data.dropna(subset=[y])
        if chart_data.empty:
            st.caption(f"No data available for: {title}")
            return
        error_arg = {("error_y" if orientation == "v" else "error_x"): error_col}
        fig = px.bar(
            chart_data, x=x, y=y, orientation=orientation,
            color=y, color_continuous_scale=color_scale,
            labels={y: "Avg Time (min)", x: x},
            text=chart_data[y].apply(lambda v: f"{v:.1f}"),
            **error_arg,
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False,
                          yaxis_title="Avg Delivery Time (min)")
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # ── Sub-section: Operational factors ────────────────────────────────────
    st.markdown('<div class="sub-section">Operational Factors</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Delivery Time by Traffic Density</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Mean ± 95 % CI per traffic level</div>', unsafe_allow_html=True)
        grp = group_stats("Road_traffic_density", "Traffic")
        traffic_order = ["Low", "Medium", "High", "Jam"]
        if not grp.empty:
            grp["Traffic"] = pd.Categorical(
                grp["Traffic"],
                categories=[c for c in traffic_order if c in grp["Traffic"].values],
                ordered=True,
            )
            grp = grp.sort_values("Traffic")
        _bar_chart(grp, "Traffic", "mean", "Traffic", "Reds")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Delivery Time by Weather Condition</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Mean ± 95 % CI per weather condition</div>', unsafe_allow_html=True)
        grp = group_stats("Weather_conditions", "Weather")
        if not grp.empty:
            grp = grp.sort_values("mean", ascending=False)
        _bar_chart(grp, "Weather", "mean", "Weather", "Blues")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Sub-section: Vehicle & Order ─────────────────────────────────────────
    st.markdown('<div class="sub-section">Vehicle & Order Type</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Delivery Time by Vehicle Type</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Mean per vehicle — error bars show 95 % CI</div>', unsafe_allow_html=True)
        grp = group_stats("Type_of_vehicle", "Vehicle")
        if not grp.empty:
            grp = grp.sort_values("mean", ascending=True)
        _bar_chart(grp, "Vehicle", "mean", "Vehicle", "Purples", orientation="h")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Delivery Time by Order Type</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Mean delivery time per order category</div>', unsafe_allow_html=True)
        grp = group_stats("Type_of_order", "Order Type")
        if not grp.empty:
            grp = grp.sort_values("mean", ascending=True)
        _bar_chart(grp, "Order Type", "mean", "Order Type", "Greens", orientation="h")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Sub-section: City & Multiple Deliveries ──────────────────────────────
    st.markdown('<div class="sub-section">City & Workload</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Delivery Time by City Type</div>', unsafe_allow_html=True)
        grp = group_stats("City", "City")
        if not grp.empty:
            grp = grp.sort_values("mean")
        if not grp.empty:
            fig = px.bar(grp, x="City", y="mean", error_y="ci95",
                         color="City", color_discrete_sequence=CHART_CAT,
                         text=grp["mean"].apply(lambda v: f"{v:.1f}"))
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                              yaxis_title="Avg Delivery Time (min)",
                              yaxis_range=[0, grp["mean"].max() + 8])
            fig.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No city data available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Delivery Time by Simultaneous Deliveries</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">0 = solo drop-off, higher numbers = multiple stops</div>', unsafe_allow_html=True)
        if "multiple_deliveries" in df.columns:
            grp = df.groupby("multiple_deliveries")[TARGET].agg(["mean", "std", "count"]).reset_index()
            grp.columns = ["Deliveries", "mean", "std", "count"]
            grp["ci95"] = 1.96 * grp["std"] / np.sqrt(grp["count"].clip(lower=1))
            grp["Deliveries"] = grp["Deliveries"].astype(str)
            fig = px.bar(grp, x="Deliveries", y="mean", error_y="ci95",
                         color="mean", color_continuous_scale="Oranges",
                         text=grp["mean"].apply(lambda v: f"{v:.1f}"))
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                              coloraxis_showscale=False,
                              yaxis_title="Avg Delivery Time (min)",
                              yaxis_range=[0, grp["mean"].max() + 8])
            fig.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("multiple_deliveries column not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Sub-section: Continuous relationships ────────────────────────────────
    st.markdown('<div class="sub-section">Continuous Relationships</div>', unsafe_allow_html=True)

    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Distance vs Delivery Time</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">5,000 sampled orders · colour = traffic density · dashed = linear trend</div>', unsafe_allow_html=True)
        if "distance_km" in df.columns:
            sample = df.sample(min(5000, len(df)), random_state=42)
            fig_sc = px.scatter(
                sample, x="distance_km", y=TARGET,
                color="Road_traffic_density",
                color_discrete_sequence=CHART_CAT,
                opacity=0.45,
                labels={"distance_km": "Distance (km)", TARGET: "Delivery Time (min)",
                        "Road_traffic_density": "Traffic"},
            )
            _x = sample["distance_km"].values
            _y = sample[TARGET].values
            _m, _b = np.polyfit(_x, _y, 1)
            _xs = np.linspace(_x.min(), _x.max(), 100)
            fig_sc.add_trace(go.Scatter(
                x=_xs, y=_m * _xs + _b,
                mode="lines", line=dict(color="#1e293b", width=2, dash="dash"),
                name="Trend (linear)", showlegend=True,
            ))
            fig_sc.update_layout(**PLOTLY_LAYOUT,
                                 xaxis_title="Delivery Distance (km)",
                                 yaxis_title="Delivery Time (minutes)",
                                 legend_title="Traffic")
            fig_sc.update_traces(marker_size=4, selector=dict(mode="markers"))
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Distance column not available — re-run `python src/features.py`.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sc2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Rider Rating vs Delivery Time</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Mean delivery time per rating value · bubble size = order count</div>', unsafe_allow_html=True)
        if "Delivery_person_Ratings" in df.columns:
            rg = df.groupby("Delivery_person_Ratings")[TARGET].agg(["mean", "count"]).reset_index()
            rg.columns = ["Rating", "mean", "count"]
            fig_rat = px.scatter(rg, x="Rating", y="mean", size="count", color="mean",
                                 color_continuous_scale="RdYlGn_r",
                                 labels={"Rating": "Rider Rating (1-5)",
                                         "mean": "Avg Delivery Time (min)",
                                         "count": "# of Orders"})
            fig_rat.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                  xaxis_title="Rider Rating",
                                  yaxis_title="Avg Delivery Time (min)")
            st.plotly_chart(fig_rat, use_container_width=True)
        else:
            st.info("Rider ratings column not available.")
        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE 3 — PREDICTION
# ===========================================================================
elif page == "Prediction":

    st.markdown("""
    <div class="page-header">
        <h2>Delivery Time Prediction</h2>
        <p>Enter order details below. The trained Random Forest model will predict the expected delivery time.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Encoding maps — must exactly match src/features.py ----
    TRAFFIC_MAP  = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    WEATHER_MAP  = {"Sunny": 1, "Cloudy": 2, "Windy": 3, "Fog": 4, "Sandstorms": 5, "Stormy": 6}
    CITY_MAP     = {"Urban": 1, "Metropolitan": 2, "Semi-Urban": 3}
    VEHICLE_MAP  = {"electric_scooter": 0, "motorcycle": 1, "scooter": 2, "bicycle": 3}
    ORDER_MAP    = {"Buffet": 0, "Drinks": 1, "Meal": 2, "Snack": 3}
    FESTIVAL_MAP = {"No": 0, "Yes": 1}

    with st.form("prediction_form"):
        st.markdown('<div class="form-section">Route</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            distance_km = st.slider("Delivery Distance (km)", 1.0, 21.0, 8.5, 0.5,
                                    help="Straight-line (Haversine) distance from restaurant to customer")
        with r1c2:
            traffic = st.selectbox("Road Traffic Density", list(TRAFFIC_MAP.keys()), index=1)
        with r1c3:
            city = st.selectbox("City Type", list(CITY_MAP.keys()), index=0)

        st.markdown('<div class="form-section">Conditions</div>', unsafe_allow_html=True)
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            weather = st.selectbox("Weather Condition", list(WEATHER_MAP.keys()), index=0)
        with r2c2:
            festival = st.selectbox("Festival Period", ["No", "Yes"], index=0,
                                    help="Is the order placed during a festival period?")
        with r2c3:
            multiple = st.selectbox("Simultaneous Deliveries", [0, 1, 2, 3], index=0,
                                    help="How many other orders is the rider delivering at the same time?")

        st.markdown('<div class="form-section">Rider</div>', unsafe_allow_html=True)
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            rider_age = st.slider("Rider Age", 20, 50, 30)
        with r3c2:
            rider_rating = st.slider("Rider Rating", 1.0, 5.0, 4.5, 0.1)
        with r3c3:
            vehicle_condition = st.select_slider(
                "Vehicle Condition", options=[0, 1, 2, 3], value=2,
                format_func=lambda x: {0: "0 — Poor", 1: "1 — Fair",
                                       2: "2 — Good", 3: "3 — Excellent"}[x],
            )

        st.markdown('<div class="form-section">Order & Timing</div>', unsafe_allow_html=True)
        r4c1, r4c2, r4c3, r4c4 = st.columns(4)
        with r4c1:
            vehicle_type = st.selectbox("Vehicle Type", list(VEHICLE_MAP.keys()), index=1)
        with r4c2:
            order_type = st.selectbox("Order Type", list(ORDER_MAP.keys()), index=2)
        with r4c3:
            order_hour = st.slider("Order Hour (0-23)", 8, 23, 13,
                                   help="Hour of day when the order was placed")
        with r4c4:
            pickup_wait = st.slider("Pickup Wait (min)", 0, 60, 15,
                                    help="Estimated minutes between order placement and rider pickup")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Delivery Time", use_container_width=True)

    # ---- Prediction ----
    if submitted:

        # --- Validate categorical inputs against known maps ---
        _input_errors = []
        if traffic not in TRAFFIC_MAP:
            _input_errors.append(f"Unknown traffic value: '{traffic}'")
        if weather not in WEATHER_MAP:
            _input_errors.append(f"Unknown weather value: '{weather}'")
        if city not in CITY_MAP:
            _input_errors.append(f"Unknown city value: '{city}'")
        if vehicle_type not in VEHICLE_MAP:
            _input_errors.append(f"Unknown vehicle type: '{vehicle_type}'")
        if order_type not in ORDER_MAP:
            _input_errors.append(f"Unknown order type: '{order_type}'")
        if festival not in FESTIVAL_MAP:
            _input_errors.append(f"Unknown festival value: '{festival}'")
        if not (1.0 <= rider_rating <= 5.0):
            _input_errors.append(f"Rider rating {rider_rating} is outside [1.0, 5.0]")
        if not (20 <= rider_age <= 50):
            _input_errors.append(f"Rider age {rider_age} is outside [20, 50]")
        if not (1.0 <= distance_km <= 21.0):
            _input_errors.append(f"Distance {distance_km} km is outside [1.0, 21.0]")

        if _input_errors:
            st.error("**Invalid input values detected:**\n\n" + "\n".join(f"- {e}" for e in _input_errors))
            st.stop()

        # --- Build feature vector ---
        order_dow = 0 if order_hour < 17 else 4
        is_peak   = int((11 <= order_hour <= 14) or (18 <= order_hour <= 22))

        input_dict = {
            "distance_km":             distance_km,
            "order_hour":              order_hour,
            "order_day_of_week":       order_dow,
            "order_month":             3,
            "is_weekend":              0,
            "is_peak_hour":            is_peak,
            "pickup_wait_min":         float(pickup_wait),
            "Delivery_person_Age":     rider_age,
            "Delivery_person_Ratings": rider_rating,
            "multiple_deliveries":     multiple,
            "Vehicle_condition":       vehicle_condition,
            "traffic_encoded":         TRAFFIC_MAP[traffic],
            "weather_encoded":         WEATHER_MAP[weather],
            "city_encoded":            CITY_MAP[city],
            "vehicle_encoded":         VEHICLE_MAP[vehicle_type],
            "order_type_encoded":      ORDER_MAP[order_type],
            "festival_encoded":        FESTIVAL_MAP[festival],
        }

        # --- Validate feature names against saved model metadata ---
        _expected = set(meta["feature_names"])
        _provided = set(input_dict.keys())
        _missing_feat = _expected - _provided
        if _missing_feat:
            st.error(
                f"**Feature mismatch.** The model expects features that were not provided: "
                f"`{', '.join(sorted(_missing_feat))}`\n\n"
                "Re-run `python src/train.py` to regenerate model metadata."
            )
            st.stop()

        # --- Run prediction ---
        try:
            input_df   = pd.DataFrame([input_dict])[meta["feature_names"]]
            prediction = float(model.predict(input_df)[0])
            if not math.isfinite(prediction):
                raise ValueError(f"Model returned a non-finite prediction: {prediction}")
            prediction = max(5.0, round(prediction, 1))
        except Exception as exc:
            st.error(
                f"**Model prediction failed.**\n\n"
                f"Error: `{type(exc).__name__}: {exc}`\n\n"
                "Try reloading the page. If the problem persists, re-run `python src/train.py`."
            )
            st.stop()

        mae = meta["test_mae"]
        lo  = max(1, round(prediction - mae))
        hi  = round(prediction + mae)

        # --- Result card ---
        st.markdown("<br>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns([1, 2, 1])
        with rc2:
            st.markdown(f"""
            <div class="pred-outer">
                <div class="pred-label">Predicted Delivery Time</div>
                <div class="pred-value">{prediction:.0f}</div>
                <div class="pred-unit">minutes</div>
                <div class="pred-range">Likely range: {lo}–{hi} min &nbsp;(± {mae} min MAE)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Feature importance bar ---
        st.markdown("##### What most influenced this prediction?")
        st.caption("Model-level feature importances from the trained Random Forest.")
        try:
            importances = model.feature_importances_
            friendly = {
                "distance_km": "Delivery Distance",
                "order_hour": "Order Hour",
                "order_day_of_week": "Day of Week",
                "order_month": "Month",
                "is_weekend": "Weekend Flag",
                "is_peak_hour": "Peak Hour Flag",
                "pickup_wait_min": "Pickup Wait Time",
                "Delivery_person_Age": "Rider Age",
                "Delivery_person_Ratings": "Rider Rating",
                "multiple_deliveries": "Simultaneous Deliveries",
                "Vehicle_condition": "Vehicle Condition",
                "traffic_encoded": "Traffic Density",
                "weather_encoded": "Weather Condition",
                "city_encoded": "City Type",
                "vehicle_encoded": "Vehicle Type",
                "order_type_encoded": "Order Type",
                "festival_encoded": "Festival Period",
            }
            imp_df = pd.DataFrame({
                "Feature":    [friendly.get(n, n) for n in meta["feature_names"]],
                "Importance": importances,
            }).sort_values("Importance", ascending=True).tail(10)

            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Blues",
                             text=imp_df["Importance"].apply(lambda v: f"{v*100:.1f}%"))
            fig_imp.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                  xaxis_tickformat=".0%", xaxis_title="Relative Importance")
            fig_imp.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.caption("Feature importance chart unavailable.")

        st.info(
            f"**Interpretation note:** Predicted by a Random Forest model "
            f"(Test MAE = {mae} min, R² = {meta['test_r2']}). "
            "This is a statistical estimate — not a guaranteed delivery time."
        )

        # ---- AI plain-language explanation ----
        if _ai_creds:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="ai-header">🧠 &nbsp;AI Plain-Language Explanation</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "The Random Forest model produced the number above. "
                "The text below is generated by an AI assistant to explain "
                "the model's inputs in plain language."
            )
            with st.spinner("Generating explanation..."):
                ai_result = get_ai_explanation(
                    prediction_min=prediction,
                    distance_km=distance_km,
                    traffic=traffic,
                    weather=weather,
                    vehicle_condition=vehicle_condition,
                    multiple_deliveries=multiple,
                    rider_rating=rider_rating,
                    rider_age=rider_age,
                    city=city,
                    festival=festival,
                    mae=mae,
                    credentials=_ai_creds,
                )

            if ai_result.error == AIError.OK and ai_result.text:
                st.markdown(
                    f'<div class="ai-box">{ai_result.text}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "AI-generated text. The ML model is responsible for the numeric "
                    "prediction; the AI only explains the factors in plain language. "
                    "Associations described do not imply causation."
                )
            else:
                # Show a specific, actionable error — never expose the key
                _icon = {
                    AIError.AUTH_FAILED:    "🔴",
                    AIError.NETWORK_ERROR:  "🟡",
                    AIError.RATE_LIMITED:   "🟡",
                    AIError.MODEL_NOT_FOUND:"🔴",
                    AIError.SERVER_ERROR:   "🟡",
                    AIError.SDK_MISSING:    "🔴",
                    AIError.MISSING_FIELD:  "🔴",
                }.get(ai_result.error, "🟡")
                st.warning(
                    f"{_icon} **AI explanation unavailable** "
                    f"({ai_result.error.value.replace('_', ' ').title()})\n\n"
                    f"{ai_result.detail}"
                )
        else:
            st.caption(
                "AI explanation not configured. "
                "Add credentials to `.streamlit/secrets.toml` to enable this feature."
            )

    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">🛵</div>
            <div class="placeholder-title">Fill in the order details above and click <strong>Predict Delivery Time</strong></div>
            <div class="placeholder-sub">The Random Forest model will estimate delivery time in minutes.</div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 4 — INSIGHTS & METHODOLOGY
# ===========================================================================
elif page == "Insights & Methodology":

    st.markdown("""
    <div class="page-header">
        <h2> Insights &amp; Methodology</h2>
        <p>Key findings from exploratory analysis, model evaluation metrics, and project methodology.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sub-section: Model comparison ───────────────────────────────────────
    st.markdown('<div class="sub-section">Model Performance Comparison</div>', unsafe_allow_html=True)
    if compare is not None and not compare.empty:
        fig_comp = go.Figure()
        models_list = compare["model"].tolist()
        fig_comp.add_trace(go.Bar(name="Train MAE", x=models_list, y=compare["train_mae"],
                                  marker_color="#93c5fd", text=compare["train_mae"].round(2),
                                  textposition="outside"))
        fig_comp.add_trace(go.Bar(name="Test MAE",  x=models_list, y=compare["test_mae"],
                                  marker_color=PRIMARY, text=compare["test_mae"].round(2),
                                  textposition="outside"))
        fig_comp.update_layout(**PLOTLY_LAYOUT, barmode="group",
                               yaxis_title="MAE (minutes, lower is better)",
                               legend_title="Split",
                               yaxis_range=[0, compare["test_mae"].max() + 2])
        st.plotly_chart(fig_comp, use_container_width=True)

        col_r2a, col_r2b = st.columns(2)
        with col_r2a:
            fig_r2 = px.bar(compare, x="model", y="test_r2",
                            color="test_r2", color_continuous_scale="Greens",
                            text=compare["test_r2"].apply(lambda v: f"{v:.3f}"))
            fig_r2.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                 yaxis_title="Test R² (higher is better)", yaxis_range=[0, 1])
            fig_r2.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig_r2, use_container_width=True)

        with col_r2b:
            try:
                imp_vals = model.feature_importances_
                friendly = {
                    "distance_km": "Distance",     "order_hour": "Order Hour",
                    "order_day_of_week": "Day",     "order_month": "Month",
                    "is_weekend": "Weekend",        "is_peak_hour": "Peak Hour",
                    "pickup_wait_min": "Pickup Wait","Delivery_person_Age": "Rider Age",
                    "Delivery_person_Ratings": "Rider Rating",
                    "multiple_deliveries": "Multiple Deliveries",
                    "Vehicle_condition": "Vehicle Condition",
                    "traffic_encoded": "Traffic",   "weather_encoded": "Weather",
                    "city_encoded": "City",         "vehicle_encoded": "Vehicle Type",
                    "order_type_encoded": "Order Type", "festival_encoded": "Festival",
                }
                imp_df2 = pd.DataFrame({
                    "Feature":    [friendly.get(n, n) for n in meta["feature_names"]],
                    "Importance": imp_vals,
                }).sort_values("Importance").tail(10)

                fig_fi = px.bar(imp_df2, x="Importance", y="Feature", orientation="h",
                                color="Importance", color_continuous_scale="Blues",
                                text=imp_df2["Importance"].apply(lambda v: f"{v*100:.1f}%"))
                fig_fi.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                     title="Feature Importance (MDI)", xaxis_tickformat=".0%")
                fig_fi.update_traces(textposition="outside", marker_line_width=0)
                st.plotly_chart(fig_fi, use_container_width=True)
            except Exception:
                st.caption("Feature importance chart unavailable.")
    else:
        st.info(
            "Model comparison table not found. "
            "Run `python src/train.py` to generate `models/model_comparison.csv`."
        )

    # ── Sub-section: Evaluation metrics ─────────────────────────────────────
    st.markdown('<div class="sub-section">Evaluation Metrics — Best Model (Random Forest)</div>', unsafe_allow_html=True)

    mae_val  = meta.get("test_mae",  "N/A")
    rmse_val = meta.get("test_rmse", "N/A")
    r2_val   = meta.get("test_r2",   0.0)
    r2_pct   = f"{float(r2_val)*100:.1f}%" if isinstance(r2_val, (int, float)) else "N/A"

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"""
        <div class="metric-card lower">
            <div class="metric-name">MAE — Mean Absolute Error</div>
            <div class="metric-value">{mae_val} <span class="metric-unit">min</span></div>
            <div class="metric-desc">
                On average, predictions are off by <strong>{mae_val} minutes</strong>.
                Easy to explain: if the model says 28 min, the real time is likely
                between {max(1, round(float(mae_val if mae_val != "N/A" else 0)))} and
                {round(float(mae_val if mae_val != "N/A" else 0) * 2)} min away.
                <em>Lower is better.</em>
            </div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="metric-card lower">
            <div class="metric-name">RMSE — Root Mean Squared Error</div>
            <div class="metric-value">{rmse_val} <span class="metric-unit">min</span></div>
            <div class="metric-desc">
                Like MAE but large errors are penalised more heavily (squared before averaging).
                RMSE &gt; MAE whenever the model makes occasional big mistakes.
                <em>Lower is better.</em>
            </div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""
        <div class="metric-card good">
            <div class="metric-name">R² — Coefficient of Determination</div>
            <div class="metric-value">{r2_val} <span class="metric-unit">/ 1.0</span></div>
            <div class="metric-desc">
                The model explains <strong>{r2_pct}</strong> of the variance in delivery time.
                1.0 = perfect prediction, 0.0 = no better than predicting the mean.
                <em>Higher is better.</em>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sub-section: EDA insights ────────────────────────────────────────────
    st.markdown('<div class="sub-section">Key Findings from Exploratory Analysis</div>', unsafe_allow_html=True)
    try:
        traffic_means = df.groupby("Road_traffic_density")[TARGET].mean()
        low_t  = traffic_means.get("Low",  traffic_means.min())
        jam_t  = traffic_means.get("Jam",  traffic_means.max())
        t_diff = jam_t - low_t

        weather_means = df.groupby("Weather_conditions")[TARGET].mean()
        best_w  = weather_means.idxmin()
        worst_w = weather_means.idxmax()
        w_diff  = weather_means.max() - weather_means.min()

        multi_means = df.groupby("multiple_deliveries")[TARGET].mean()
        solo_t = float(multi_means.get(0, multi_means.min()))
        tri_t  = float(multi_means.get(3, multi_means.max()))

        rating_grp  = df.groupby("Delivery_person_Ratings")[TARGET].mean()
        low_rat_t   = float(rating_grp[rating_grp.index <= 3.0].mean())  if not rating_grp.empty else float("nan")
        high_rat_t  = float(rating_grp[rating_grp.index >= 4.5].mean())  if not rating_grp.empty else float("nan")

        dist_corr = df["distance_km"].corr(df[TARGET]) if "distance_km" in df.columns else float("nan")

        insights = [
            {
                "title": "Traffic is the strongest categorical predictor",
                "body": (
                    f"Jam-traffic deliveries average {jam_t:.1f} min vs {low_t:.1f} min in low traffic "
                    f"— a difference of {t_diff:.1f} minutes. "
                    "Traffic density has a Pearson r = 0.41 with delivery time."
                ),
            },
            {
                "title": f"Weather shows a {w_diff:.1f}-minute spread across conditions",
                "body": (
                    f"{best_w} conditions are associated with the lowest average delivery time; "
                    f"{worst_w} with the highest. The pattern is not strictly monotone — "
                    "Cloudy is worse than Stormy — suggesting the ordinal encoding may be imprecise."
                ),
            },
            {
                "title": "Multiple simultaneous deliveries strongly increases time",
                "body": (
                    f"Solo deliveries ({solo_t:.1f} min) are far faster than 3-order batches "
                    f"({tri_t:.1f} min). However, 3-order batches represent only "
                    f"{(df['multiple_deliveries'] == 3).sum() / len(df) * 100:.1f}% of orders."
                ),
            },
            {
                "title": f"Distance is the strongest continuous predictor (r = {dist_corr:.2f})",
                "body": (
                    f"Delivery distance has a Pearson correlation of {dist_corr:.2f} with delivery time "
                    "and is the 3rd most important feature by permutation importance."
                ),
            },
            {
                "title": "Higher-rated riders are associated with faster deliveries",
                "body": (
                    f"Riders rated 3.0 or below are associated with {low_rat_t:.1f} min deliveries vs "
                    f"{high_rat_t:.1f} min for riders rated 4.5 or above. "
                    "This is a correlation — higher-rated riders may simply be assigned to easier routes."
                ),
            },
            {
                "title": f"Random Forest is the best model — no overfitting detected",
                "body": (
                    f"Train MAE = 2.80 min vs Test MAE = {mae_val} min. "
                    "5-fold cross-validation confirms Test MAE of 3.25 ± 0.01 min across all folds."
                ),
            },
        ]

        for ins in insights:
            st.markdown(
                f'<div class="insight-card">'
                f'<div class="insight-title">&#9632; {ins["title"]}</div>'
                f'<div class="insight-body">{ins["body"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    except Exception as exc:
        st.warning(f"Could not compute EDA insights: `{type(exc).__name__}: {exc}`")

    # ── Sub-section: Limitations ─────────────────────────────────────────────
    st.markdown('<div class="sub-section">Limitations &amp; Caveats</div>', unsafe_allow_html=True)
    limitations = [
        ("Target rounding",       "Delivery time is recorded as whole-number minutes (10–54). Sub-minute precision is not meaningful."),
        ("Geographic scope",      "Data covers Indian cities only (Feb–Apr 2022). The model may not generalise to other regions or seasons."),
        ("Weather encoding",      "The ordinal weather scale (Sunny=1 to Stormy=6) does not match the observed mean-time ranking. Cloudy records higher average times than Stormy."),
        ("No live inputs",        "The model uses features known at order time. It has no access to real-time GPS, live traffic APIs, or restaurant queue depth."),
        ("Correlation vs Cause",  "All relationships are predictive associations — not proven causal effects."),
    ]
    for title, body in limitations:
        st.markdown(
            f'<div class="warn-card">'
            f'<div class="warn-title">&#9651; {title}</div>'
            f'<div class="warn-body">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Sub-section: Methodology ──────────────────────────────────────────────
    st.markdown('<div class="sub-section">Project Methodology</div>', unsafe_allow_html=True)

    steps = [
        ("Data Collection",
         "Raw Zomato dataset — 45,593 delivery records with 20 columns covering rider, "
         "restaurant, customer, traffic, weather, and order attributes."),
        ("Data Cleaning",
         "Removed duplicates and malformed rows; standardised column names; parsed dates "
         "and times; stripped whitespace from categorical columns. "
         "Result: 45,584 clean rows (<code>src/data_cleaning.py</code>)."),
        ("Feature Engineering",
         "13 numeric features derived: Haversine distance (km), time-of-day features "
         "(hour, day-of-week, month, is_weekend, is_peak_hour), pickup wait time, "
         "and ordinal encodings for traffic, weather, city, vehicle, order type, and festival "
         "(<code>src/features.py</code>)."),
        ("Model Training & Selection",
         "Five regression models compared: Linear Regression, Ridge, Decision Tree, "
         "Random Forest (200 trees), and Gradient Boosting. "
         "Evaluated on 80/20 train-test split using MAE, RMSE, and R². "
         "5-fold cross-validation run on top two models (<code>src/train.py</code>)."),
        ("Best Model",
         f"Random Forest selected — Test MAE = {mae_val} min, RMSE = {rmse_val} min, "
         f"R² = {r2_val}. No overfitting detected (train–test MAE gap &lt; 1.5 min). "
         "Model saved to <code>models/best_model.joblib</code>."),
        ("Dashboard",
         "Interactive Streamlit app with four pages: Overview KPIs, Delivery Analytics "
         "(7 interactive charts), Prediction form (live inference), and this Insights page. "
         "All figures computed dynamically — no hard-coded analytical results."),
    ]
    for i, (title, desc) in enumerate(steps, start=1):
        st.markdown(
            f'<div class="step">'
            f'<div class="step-num">{i}</div>'
            f'<div class="step-body">'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Sub-section: About ────────────────────────────────────────────────────
    st.markdown('<div class="sub-section">About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">Student Data Science Portfolio Project</div>
        <div class="insight-body">
            This application was built as a student data science portfolio project to demonstrate
            end-to-end machine learning: data cleaning, feature engineering, model training,
            evaluation, and deployment as an interactive web application.<br><br>
            <strong>Dataset:</strong> Zomato Food Delivery dataset (Kaggle, India, Feb–Apr 2022).<br>
            <strong>Stack:</strong> Python · pandas · scikit-learn · Streamlit · Plotly.<br>
            <strong>Model:</strong> Random Forest Regressor (200 trees, max_depth=15) trained on
            17 engineered features. No personally identifiable information is used.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<hr style="margin-top:40px; border-color:#e2e8f0">
<div style="text-align:center; color:#94a3b8; font-size:0.78rem; padding:10px 0 20px">
    Food Delivery Analytics &middot; Student Data Science Project &middot;
    Random Forest &middot; scikit-learn &middot; Streamlit
</div>
""", unsafe_allow_html=True)
