"""
CardioRisk AI — UI Components
==============================
All reusable Streamlit/Matplotlib rendering functions.
app.py calls these; no business logic lives here.
"""

from __future__ import annotations

import io
import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from src.config import (
    APP_NAME, APP_VERSION, MODEL_VERSION,
    PLOT_BG, PLOT_BG2, PLOT_BLUE, PLOT_CYN, PLOT_GRN,
    PLOT_GRID, PLOT_ORG, PLOT_PRP, PLOT_RED, PLOT_T1, PLOT_T2, PLOT_T3,
    RISK_HIGH_THRESHOLD,
    RISK_LABELS, RISK_LOW_THRESHOLD, RISK_MED_THRESHOLD, RISK_NOTES,
    SIDEBAR_CP, SIDEBAR_CP_MAP,
    SIDEBAR_ECG, SIDEBAR_ECG_MAP,
    SIDEBAR_SLOPE, SIDEBAR_SLOPE_MAP,
    SIDEBAR_THAL, SIDEBAR_THAL_MAP,
    AGE_RANGE, BP_RANGE, CHOL_RANGE, HR_RANGE, OLDPEAK_RANGE, CA_RANGE,
    FEATURE_LABELS,
)


# ── CSS injection ─────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #040810 !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', 'SF Pro Display', system-ui, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #070d1c !important;
        border-right: 1px solid #0f2040;
    }
    /* ── Card ── */
    .card {
        background: linear-gradient(135deg, #0a1428 0%, #0d1b35 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,.4);
    }
    .card-sm {
        background: #0a1428;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    /* ── Hero metric ── */
    .hero-metric {
        font-size: 3.8rem;
        font-weight: 800;
        letter-spacing: -2px;
        line-height: 1;
    }
    .hero-low  { color: #10b981; }
    .hero-med  { color: #f59e0b; }
    .hero-high { color: #ef4444; }
    /* ── Risk badge ── */
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: .85rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .badge-low  { background: rgba(16,185,129,.15); color: #10b981; border: 1px solid #10b981; }
    .badge-med  { background: rgba(245,158,11,.15); color: #f59e0b; border: 1px solid #f59e0b; }
    .badge-high { background: rgba(239,68,68,.15);  color: #ef4444; border: 1px solid #ef4444; }
    /* ── Disclaimer banner ── */
    .disclaimer {
        background: rgba(245,158,11,.08);
        border: 1px solid rgba(245,158,11,.3);
        border-radius: 10px;
        padding: 12px 18px;
        font-size: .78rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    /* ── Divider ── */
    hr.section { border: none; border-top: 1px solid #0f2040; margin: 28px 0; }
    /* ── Streamlit overrides ── */
    .stSlider > div { color: #94a3b8 !important; }
    div[data-testid="metric-container"] {
        background: #0a1428;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 12px 16px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 28px;
        transition: all .2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(37,99,235,.4); }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render all sidebar inputs and return a dict of raw widget values."""
    with st.sidebar:
        st.markdown(f"## 🫀 {APP_NAME}")
        st.caption(f"v{APP_VERSION} · Model v{MODEL_VERSION}")
        st.markdown("---")

        st.markdown("#### 👤 Demographics")
        age = st.slider("Age (years)", *AGE_RANGE, 55,
                        help="Patient age in years")
        sex = st.radio("Biological Sex", ["Male", "Female"],
                       horizontal=True)

        st.markdown("#### 🩺 Clinical Measurements")
        trestbps = st.slider("Resting BP (mmHg)", *BP_RANGE, 130,
                             help="Resting systolic blood pressure on hospital admission")
        chol     = st.slider("Cholesterol (mg/dL)", *CHOL_RANGE, 240,
                             help="Serum cholesterol — target < 200 mg/dL")
        thalch   = st.slider("Max Heart Rate (bpm)", *HR_RANGE, 150,
                             help="Maximum heart rate achieved during exercise test")
        oldpeak  = st.slider("ST Depression", *OLDPEAK_RANGE, 1.0, step=0.1,
                             help="ST depression induced by exercise relative to rest")
        ca       = st.slider("Fluoroscopy Vessels (0–3)", *CA_RANGE, 0,
                             help="Number of major vessels coloured by fluoroscopy")

        st.markdown("#### 🔬 Diagnostic")
        cp      = st.selectbox("Chest Pain Type", SIDEBAR_CP,
                               help="Type of chest pain reported by the patient")
        restecg = st.selectbox("Resting ECG", SIDEBAR_ECG,
                               help="Resting electrocardiographic results")
        slope   = st.selectbox("ST Slope", SIDEBAR_SLOPE,
                               help="Slope of the peak exercise ST segment")
        thal    = st.selectbox("Thalassemia", SIDEBAR_THAL,
                               help="Thallium stress test result")

        st.markdown("#### ✅ Binary Flags")
        fbs   = st.checkbox("Fasting Blood Sugar > 120 mg/dL",
                            help="True if fasting blood glucose > 120 mg/dL")
        exang = st.checkbox("Exercise-Induced Angina",
                            help="Angina symptoms during exercise test")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            predict_btn = st.button("🔍 Analyse", use_container_width=True)
        with col2:
            reset_btn = st.button("↺ Reset", use_container_width=True)

        st.markdown("""
        <div class='disclaimer'>
        ⚠️ <strong>Research use only.</strong><br>
        Not a substitute for professional medical advice, diagnosis, or treatment.
        </div>""", unsafe_allow_html=True)

    return dict(
        age=age, sex=sex, cp=SIDEBAR_CP_MAP[cp],
        cp_label=cp, trestbps=trestbps, chol=chol,
        fbs=fbs, restecg=SIDEBAR_ECG_MAP[restecg],
        restecg_label=restecg, thalch=thalch,
        exang=exang, oldpeak=oldpeak,
        slope=SIDEBAR_SLOPE_MAP[slope], slope_label=slope,
        ca=ca, thal=SIDEBAR_THAL_MAP[thal], thal_label=thal,
        predict_btn=predict_btn, reset_btn=reset_btn,
    )


# ── Hero / Header ─────────────────────────────────────────────────────────────

def render_hero() -> None:
    st.markdown("""
    <div class="card" style="text-align:center; padding: 40px 28px;">
        <div style="font-size:3rem; margin-bottom:8px;">🫀</div>
        <h1 style="font-size:2.4rem; font-weight:800; margin:0; color:#e2e8f0;
                   letter-spacing:-1px;">CardioRisk AI</h1>
        <p style="color:#64748b; margin:10px 0 0; font-size:1.05rem;">
            Clinical-grade cardiovascular risk stratification · UCI 4-site dataset · XGBoost + LR Ensemble
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Model Performance Banner ──────────────────────────────────────────────────

def render_model_metrics(bundle) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGBoost AUC",   f"{bundle.xgb_auc:.3f}", help="Test-set ROC-AUC")
    c2.metric("Logistic AUC",  f"{bundle.lr_auc:.3f}",  help="Test-set ROC-AUC")
    c3.metric("5-Fold CV AUC", f"{bundle.cv_auc:.3f}",  help="Cross-validated AUC (anti-leakage pipeline)")
    c4.metric("Training Data", "920 patients",           help="4-site UCI Heart Disease dataset")


# ── Risk Result Panel ─────────────────────────────────────────────────────────

def render_result(result) -> None:
    p = result.prob_ensemble
    if p < RISK_LOW_THRESHOLD:
        tier, cls, badge_cls = "low",  "hero-low",  "badge-low"
    elif p < RISK_MED_THRESHOLD:
        tier, cls, badge_cls = "med",  "hero-med",  "badge-med"
    else:
        tier, cls, badge_cls = "high", "hero-high", "badge-high"

    label = RISK_LABELS[tier]
    note  = RISK_NOTES[tier]

    st.markdown(f"""
    <div class="card" style="text-align:center;">
        <div class="hero-metric {cls}">{result.risk_pct:.1f}%</div>
        <div style="margin:14px 0 8px;">
            <span class="risk-badge {badge_cls}">{label}</span>
        </div>
        <p style="color:#94a3b8; font-size:.92rem; max-width:520px;
                  margin:0 auto; line-height:1.7;">{note}</p>
        <hr style="border-color:#1e3a5f; margin:20px 0;">
        <div style="display:flex; justify-content:center; gap:40px; flex-wrap:wrap;">
            <div><div style="color:#64748b; font-size:.78rem; text-transform:uppercase;
                             letter-spacing:.08em;">XGBoost</div>
                 <div style="font-size:1.4rem; font-weight:700; color:#3b82f6;">
                     {result.prob_xgb*100:.1f}%</div></div>
            <div><div style="color:#64748b; font-size:.78rem; text-transform:uppercase;
                             letter-spacing:.08em;">Logistic Reg.</div>
                 <div style="font-size:1.4rem; font-weight:700; color:#8b5cf6;">
                     {result.prob_lr*100:.1f}%</div></div>
            <div><div style="color:#64748b; font-size:.78rem; text-transform:uppercase;
                             letter-spacing:.08em;">Ensemble</div>
                 <div style="font-size:1.4rem; font-weight:700; color:#10b981;">
                     {result.prob_ensemble*100:.1f}%</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Risk Gauge ────────────────────────────────────────────────────────────────

def plot_gauge(prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_BG)

    # Gradient arc zones
    theta   = np.linspace(np.pi, 0, 300)
    r_outer, r_inner = 1.0, 0.55
    zones = [
        (0.00, 0.30, PLOT_GRN),
        (0.30, 0.60, PLOT_ORG),
        (0.60, 1.00, PLOT_RED),
    ]
    for lo, hi, colour in zones:
        t_lo = np.pi * (1 - hi)
        t_hi = np.pi * (1 - lo)
        t    = np.linspace(t_lo, t_hi, 100)
        x_o  = r_outer * np.cos(t)
        y_o  = r_outer * np.sin(t)
        x_i  = r_inner * np.cos(t)
        y_i  = r_inner * np.sin(t)
        ax.fill(
            np.concatenate([x_o, x_i[::-1]]),
            np.concatenate([y_o, y_i[::-1]]),
            color=colour, alpha=0.85,
        )

    # Needle
    angle = np.pi * (1 - prob)
    needle_len = 0.72
    ax.plot(
        [0, needle_len * np.cos(angle)],
        [0, needle_len * np.sin(angle)],
        color="white", lw=3, zorder=5, solid_capstyle="round",
    )
    ax.add_patch(plt.Circle((0, 0), 0.06, color="white", zorder=6))

    # Labels
    for val, label in [(0.0, "0"), (0.3, "30"), (0.6, "60"), (1.0, "100")]:
        a = np.pi * (1 - val)
        ax.text(
            1.18 * np.cos(a), 1.18 * np.sin(a), f"{label}",
            ha="center", va="center", color=PLOT_T3, fontsize=8,
        )

    # Centre score
    ax.text(0, -0.18, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color="white")
    ax.text(0, -0.36, "Cardiac Risk Score", ha="center", va="center",
            fontsize=8, color=PLOT_T2)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.55, 1.25)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(fi: dict[str, float]) -> plt.Figure:
    labels = list(fi.keys())[:10]
    values = list(fi.values())[:10]
    labels.reverse(); values.reverse()

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_BG2)

    colours = [PLOT_BLUE if v < max(values) * 0.6 else PLOT_CYN for v in values]
    bars = ax.barh(labels, values, color=colours, height=0.65, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=PLOT_T3)

    ax.set_xlabel("Importance Score", color=PLOT_T2, fontsize=9)
    ax.set_title("Top Feature Importances (XGBoost)", color="#e2e8f0",
                 fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(colors=PLOT_T2, labelsize=9)
    ax.xaxis.label.set_color(PLOT_T2)
    for spine in ax.spines.values():
        spine.set_edgecolor(PLOT_T0)
    ax.set_facecolor(PLOT_BG2)
    ax.grid(axis="x", color=PLOT_GRID, linewidth=0.6, zorder=0)
    fig.tight_layout()
    return fig


# ── ROC / PR Curves ───────────────────────────────────────────────────────────

def plot_roc_pr(bundle) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=PLOT_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(PLOT_BG2)
        for spine in ax.spines.values():
            spine.set_edgecolor(PLOT_T0)
        ax.tick_params(colors=PLOT_T2, labelsize=9)
        ax.grid(color=PLOT_GRID, linewidth=0.5)

    # ROC
    for model, name, colour, X_in in [
        (bundle.xgb, "XGBoost",       PLOT_BLUE, bundle.X_test),
        (bundle.lr,  "Logistic Reg.", PLOT_PRP,  bundle.X_test_s),
    ]:
        probs = model.predict_proba(X_in)[:, 1]
        fpr, tpr, _ = roc_curve(bundle.y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colour, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax1.plot([0, 1], [0, 1], "--", color=PLOT_T1, lw=1)
    ax1.set_xlabel("False Positive Rate", color=PLOT_T2, fontsize=9)
    ax1.set_ylabel("True Positive Rate",  color=PLOT_T2, fontsize=9)
    ax1.set_title("ROC Curve", color="#e2e8f0", fontweight="bold", fontsize=11)
    ax1.legend(fontsize=8, facecolor=PLOT_BG, labelcolor=PLOT_T3)

    # PR
    for model, name, colour, X_in in [
        (bundle.xgb, "XGBoost",       PLOT_BLUE, bundle.X_test),
        (bundle.lr,  "Logistic Reg.", PLOT_PRP,  bundle.X_test_s),
    ]:
        probs = model.predict_proba(X_in)[:, 1]
        prec, rec, _ = precision_recall_curve(bundle.y_test, probs)
        ap = average_precision_score(bundle.y_test, probs)
        ax2.plot(rec, prec, color=colour, lw=2, label=f"{name} (AP={ap:.3f})")

    ax2.set_xlabel("Recall",    color=PLOT_T2, fontsize=9)
    ax2.set_ylabel("Precision", color=PLOT_T2, fontsize=9)
    ax2.set_title("Precision–Recall Curve", color="#e2e8f0", fontweight="bold", fontsize=11)
    ax2.legend(fontsize=8, facecolor=PLOT_BG, labelcolor=PLOT_T3)

    fig.tight_layout(pad=2.0)
    return fig


# ── What-If Simulator ─────────────────────────────────────────────────────────

def render_whatif(bundle, base_input_df: pd.DataFrame, base_prob: float) -> None:
    from src.models import predict as model_predict

    st.markdown("### 🔬 What-If Simulator")
    st.caption("Adjust individual risk factors to see how they change your score.")

    cols = st.columns(3)
    adjustable = {
        "trestbps": ("Resting BP", BP_RANGE,    int(base_input_df["trestbps"].iloc[0])),
        "chol":     ("Cholesterol", CHOL_RANGE, int(base_input_df["chol"].iloc[0])),
        "thalch":   ("Max HR",      HR_RANGE,   int(base_input_df["thalch"].iloc[0])),
        "oldpeak":  ("ST Depression", (0.0, 6.2), float(round(base_input_df["oldpeak"].iloc[0], 1))),
        "ca":       ("Vessels",     (0, 3),      int(base_input_df["ca"].iloc[0])),
        "age":      ("Age",         AGE_RANGE,   int(base_input_df["age"].iloc[0])),
    }

    modified = base_input_df.copy()
    for i, (col, (label, (lo, hi), default)) in enumerate(adjustable.items()):
        with cols[i % 3]:
            step = 0.1 if isinstance(default, float) else 1
            new_val = st.slider(
                label, lo, hi, default,
                step=step,
                key=f"whatif_{col}",
            )
            modified[col] = new_val

    new_result = model_predict(bundle, modified)
    delta      = new_result.prob_ensemble - base_prob
    delta_str  = f"{'▲' if delta > 0 else '▼'} {abs(delta)*100:.1f}pp vs current"
    colour     = "#ef4444" if delta > 0 else "#10b981"

    st.markdown(f"""
    <div class="card-sm" style="text-align:center; margin-top:12px;">
        <span style="font-size:2rem; font-weight:800; color:white;">
            {new_result.risk_pct:.1f}%
        </span>
        &nbsp;
        <span style="font-size:1rem; font-weight:600; color:{colour};">
            {delta_str}
        </span>
    </div>
    """, unsafe_allow_html=True)


# ── SHAP waterfall (optional dep) ─────────────────────────────────────────────

def plot_shap_waterfall(bundle, input_df: pd.DataFrame) -> plt.Figure | None:
    try:
        import shap  # optional — gracefully absent
    except ImportError:
        return None

    explainer  = shap.TreeExplainer(bundle.xgb)
    shap_vals  = explainer(input_df[bundle.feature_cols].values.astype(np.float32))

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=PLOT_BG)
    shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
    plt.gcf().set_facecolor(PLOT_BG)
    fig = plt.gcf()
    return fig


# ── Disclaimer banner ─────────────────────────────────────────────────────────

def render_disclaimer() -> None:
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer</strong> — CardioRisk AI is intended for
    <strong>research and educational purposes only</strong>. It is not a
    certified medical device and has not been evaluated by the FDA, CE, or any
    regulatory body. Predictions must not be used as a substitute for
    professional clinical judgement. Always consult a qualified healthcare
    provider. Patient data entered here is processed locally and not stored.
    </div>
    """, unsafe_allow_html=True)


# ── Distribution charts ───────────────────────────────────────────────────────

def plot_age_dist(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_BG2)
    for target, colour, label in [(0, PLOT_GRN, "No Disease"), (1, PLOT_RED, "Disease")]:
        ax.hist(df.loc[df["target"] == target, "age"] if "target" in df.columns
                else df["age"], bins=20, color=colour, alpha=0.7, label=label, edgecolor="none")
    ax.set_xlabel("Age", color=PLOT_T2, fontsize=9)
    ax.set_ylabel("Count", color=PLOT_T2, fontsize=9)
    ax.set_title("Age Distribution by Outcome", color="#e2e8f0", fontweight="bold", fontsize=11)
    ax.tick_params(colors=PLOT_T2, labelsize=9)
    ax.legend(fontsize=8, facecolor=PLOT_BG, labelcolor=PLOT_T3)
    for spine in ax.spines.values():
        spine.set_edgecolor(PLOT_T0)
    ax.grid(color=PLOT_GRID, linewidth=0.5)
    fig.tight_layout()
    return fig
