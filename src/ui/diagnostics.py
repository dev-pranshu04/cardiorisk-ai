"""
Model registry panel + EDA dashboard row (top-of-page, always visible).
"""
import streamlit as st

from src.config import MODEL_VERSION
from src.plots import render_eda_overview


def render_model_registry(xgb_auc, cv_auc, xgb_ap, lr_auc, lr_ap) -> None:
    st.markdown('<p class="sec-hd">Model Registry</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-card primary">
        <div class="model-tag">
            <span class="model-name">XGBoost v2</span>
            <span class="model-badge">PRIMARY \u00b7 65%</span>
        </div>
        <div class="model-auc">{xgb_auc:.3f}</div>
        <div class="model-meta">AUC \u00b7 CV={cv_auc:.3f} \u00b7 AP={xgb_ap:.3f}</div>
    </div>
    <div class="model-card">
        <div class="model-tag">
            <span class="model-name">Logistic Regression</span>
            <span style="font-size:9px;color:#1e3a5f;">35%</span>
        </div>
        <div class="model-auc" style="font-size:26px;">{lr_auc:.3f}</div>
        <div class="model-meta">AUC \u00b7 AP={lr_ap:.3f} \u00b7 Interpretable</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-hd" style="margin-top:20px;">Ensemble Strategy</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-chips">
        <span class="metric-chip">XGB <strong>65%</strong></span>
        <span class="metric-chip">LR <strong>35%</strong></span>
        <span class="metric-chip">Weighted avg</span>
    </div>
    <p style="font-size:11px;color:#1e3a5f;line-height:1.7;">
        Soft-probability blending combines XGBoost's
        non-linear feature interactions with Logistic Regression's
        calibrated linear baseline for robust generalization.
    </p>
    <p style="font-size:10px;color:#1e3a5f;margin-top:12px;">
        Model version <strong>v{MODEL_VERSION}</strong>
    </p>""", unsafe_allow_html=True)


def render_clinical_guide() -> None:
    st.markdown('<p class="sec-hd">Clinical Guide</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-panel">
        <p>
            Enter the patient's clinical vitals in the left panel, then click
            <strong>Compute Cardiac Risk Score</strong> to generate the
            ensemble prediction.<br><br>
            The system outputs a calibrated probability score alongside
            per-feature importance, model diagnostics, ROC curves, and
            a calibration plot for clinical interpretation.
        </p>
        <div class="warning">
            Research &amp; educational use only. Not a certified medical device.
            Do not use for clinical diagnosis without physician oversight.
        </div>
    </div>""", unsafe_allow_html=True)


def render_dashboard_row(models: dict) -> None:
    c1, c2, c3 = st.columns([1, 1.75, 1])

    with c1:
        render_model_registry(
            models["xgb_auc"], models["cv_auc"], models["xgb_ap"],
            models["lr_auc"], models["lr_ap"],
        )

    with c2:
        st.markdown('<p class="sec-hd">Exploratory Data Analysis</p>', unsafe_allow_html=True)
        render_eda_overview(models["df_raw"], models["y"])

    with c3:
        render_clinical_guide()

    st.markdown("<hr>", unsafe_allow_html=True)
