"""
CardioRisk AI — Streamlit entry point.

Restructured per Step 1 audit: all logic split into src/ modules.
Fixes applied here:
  P0-2: train_models() is now a no-argument @st.cache_resource call.
  P0-4: load_data() handles a missing CSV with a friendly error (src/data.py).
  P1-4: prediction results persist in st.session_state, so sidebar
        changes after clicking "Compute" don't wipe the risk display.
  P1-8: global warning suppression removed; only noisy library loggers
        are silenced.
"""
import logging

import streamlit as st

from src.config import apply_matplotlib_theme
from src.models import train_models
from src.predict import compute_prediction
from src.data import load_data  # noqa: F401  (ensures cache warms via train_models)

from src.ui.styles import inject_global_styles
from src.ui.hero import inject_favicon, render_hero
from src.ui.kpi_strip import render_kpi_strip
from src.ui.diagnostics import render_dashboard_row
from src.ui.sidebar import render_sidebar
from src.ui.risk_display import (
    render_validation_warnings, render_risk_score, render_model_contributions,
    render_pdf_download, render_patient_summary_table, render_classification_report,
)
from src.ui.whatif import render_whatif_simulator
from src.ui.footer import render_footer
from src.shap_explain import render_shap_waterfall
from src.plots import (
    render_feature_importance, render_roc_curve, render_pr_curve,
    render_calibration_curve, render_confusion_matrix,
    render_threshold_sensitivity, render_chest_pain_outcome, render_max_hr_outcome,
)
from src.config import FEATS, FLABELS

# P1-8: silence noisy third-party loggers instead of global warning suppression
logging.getLogger("xgboost").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI · Clinical Decision Support",
    page_icon="\U0001fac0",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_favicon()
inject_global_styles()
apply_matplotlib_theme()

# ─────────────────────────────────────────────────────────────────────────────
# DATA + MODELS (cached)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models..."):
    models = train_models()

# ─────────────────────────────────────────────────────────────────────────────
# HERO + KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
render_hero(models["n_patients"], models["xgb_auc"])
render_kpi_strip(
    models["xgb_auc"], models["cv_auc"], models["lr_auc"],
    models["prevalence"], models["y"].sum(),
)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD ROW (Model Registry + EDA + Clinical Guide)
# ─────────────────────────────────────────────────────────────────────────────
render_dashboard_row(models)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
inputs, go = render_sidebar(models["df_raw"])

# P1-4: persist prediction across reruns triggered by sidebar changes
if go:
    st.session_state["result"] = compute_prediction(inputs, models)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION VIEW
# ─────────────────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]

    render_validation_warnings(result["inputs"])

    # ── Row 1: Risk + breakdown + feature importance ────────────────────────
    r1, r2 = st.columns([1, 1.9])
    with r1:
        render_risk_score(result)
        render_model_contributions(result)
        render_pdf_download(result, models)

    with r2:
        st.markdown('<p class="sec-hd">Feature Importance \u2014 XGBoost (Gain)</p>', unsafe_allow_html=True)
        render_feature_importance(models["xgb"], FEATS, FLABELS)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 2: SHAP explainability ───────────────────────────────────────────
    st.markdown('<p class="sec-hd">Per-Patient Explainability (SHAP)</p>', unsafe_allow_html=True)
    render_shap_waterfall(models["xgb"], result["Xi_imp"], FLABELS, FEATS)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 3: What-If Simulator ─────────────────────────────────────────────
    render_whatif_simulator(result, models)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 4: ROC + PR + Calibration + CM ───────────────────────────────────
    st.markdown('<p class="sec-hd">Model Diagnostics</p>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)

    xgb_prob_te = models["xgb"].predict_proba(models["X_test"])[:, 1]
    lr_prob_te = models["lr"].predict_proba(models["X_test_scaled"])[:, 1]

    with d1:
        render_roc_curve(models["y_test"], xgb_prob_te, lr_prob_te)
    with d2:
        render_pr_curve(models["y_test"], xgb_prob_te, lr_prob_te)
    with d3:
        render_calibration_curve(models["y_test"], xgb_prob_te, lr_prob_te)
    with d4:
        render_confusion_matrix(models["y_test"], models["xgb"], models["X_test"])

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 5: Classification Report + Threshold + Patient Table ────────────
    p1, p2 = st.columns([1.1, 1])
    with p1:
        render_classification_report(models["y_test"], models["xgb"], models["X_test"])
        st.markdown('<p class="sec-hd" style="margin-top:18px;">Threshold Sensitivity (XGBoost)</p>', unsafe_allow_html=True)
        render_threshold_sensitivity(models["y_test"], xgb_prob_te)
    with p2:
        render_patient_summary_table(result["inputs"])

else:
    # ─────────────────────────────────────────────────────────────────────────
    # EXPLORATORY VIEW (no prediction yet)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<p class="sec-hd">Exploratory Analysis</p>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        render_chest_pain_outcome(models["df_raw"])
    with e2:
        render_max_hr_outcome(models["df_raw"])

    st.markdown("""
    <div style='text-align:center;padding:28px 0 12px;font-size:13px;color:#1e3a5f;'>
        \u2190 &nbsp; Configure patient vitals and click
        <span style='color:#3b82f6;font-weight:600;'>Compute Cardiac Risk Score</span>
        to view the full diagnostic report
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
render_footer()

