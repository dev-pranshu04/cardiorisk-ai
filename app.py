
"""
CardioRisk AI — Main Application Entry Point
=============================================
This file is intentionally thin: it wires together config, data, models,
UI, and reports.  All logic lives in src/.

Fixes applied
-------------
P0-1  exang encoding typo         → src/config.py ENCODE_EXANG
P0-2  zero-value sentinel cleaning → src/data._clean()
P0-3  double session_state guard   → removed; @st.cache_resource handles it
P1-1  CSV error boundary           → try/except around load_and_prepare()
P1-2  pinned requirements          → requirements.txt
P1-3  input validation             → src/data._validate_ranges()
P1-4  missingness flags            → src/data._add_missingness_flags()
P1-6  reset button                 → sidebar ↺ Reset
P1-7  model version metadata       → ModelBundle.version / .trained
P2-9  CV leakage                   → Pipeline in src/models.train_models()
"""

from __future__ import annotations

import logging
import sys

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/dev-pranshu04/cardiorisk-ai",
        "Report a bug":"https://github.com/dev-pranshu04/cardiorisk-ai/issues",
        "About":       "CardioRisk AI v2 — cardiovascular risk stratification.",
    },
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Local imports (after page config) ────────────────────────────────────────
from src.config import APP_NAME, APP_VERSION, DATA_FILE
from src.data   import load_and_prepare, encode_input
from src.models import train_models, predict
from src.ui     import (
    inject_css, render_sidebar, render_hero,
    render_model_metrics, render_result,
    plot_gauge, plot_feature_importance, plot_roc_pr,
    render_whatif, render_disclaimer,
)
from src.reports import generate_pdf_report


# ── Cached data & model loaders ───────────────────────────────────────────────

@st.cache_data(show_spinner="📊 Loading & cleaning dataset…")
def get_data():
    """Load, clean, encode, and impute once per session."""
    return load_and_prepare()   # returns (df, X, y, sha256)


@st.cache_resource(show_spinner="🧠 Training ensemble models…")
def get_models(sha256: str):
    """
    Train models keyed on data hash — retrain only when data changes.
    The sha256 arg makes the cache key data-sensitive.
    P0-3 fix: no more if/else session_state anti-pattern.
    """
    _, X, y, _ = get_data()
    return train_models(X, y)


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()

    # ── Load data (P1-1: error boundary) ─────────────────────────────────────
    try:
        df, X, y, sha256 = get_data()
    except FileNotFoundError as exc:
        st.error(f"**Data file not found:** {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"**Failed to load data:** {exc}")
        logger.exception("Data load failed")
        st.stop()

    # ── Load / retrieve cached models ─────────────────────────────────────────
    bundle = get_models(sha256)

    # ── Sidebar inputs ─────────────────────────────────────────────────────────
    inputs = render_sidebar()

    # ── Reset ─────────────────────────────────────────────────────────────────
    if inputs["reset_btn"]:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # ── Hero ──────────────────────────────────────────────────────────────────
    render_hero()

    # ── Model metrics strip ────────────────────────────────────────────────────
    render_model_metrics(bundle)

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if inputs["predict_btn"]:
        try:
            input_df = encode_input(
                age      = inputs["age"],
                sex      = inputs["sex"],
                cp       = inputs["cp_label"],
                trestbps = inputs["trestbps"],
                chol     = inputs["chol"],
                fbs      = inputs["fbs"],
                restecg  = inputs["restecg_label"],
                thalch   = inputs["thalch"],
                exang    = inputs["exang"],
                oldpeak  = inputs["oldpeak"],
                slope    = inputs["slope_label"],
                ca       = inputs["ca"],
                thal     = inputs["thal_label"],
            )
        except ValueError as exc:
            st.error(f"**Input validation error:** {exc}")
            st.stop()

        result = predict(bundle, input_df)
        st.session_state["last_result"]   = result
        st.session_state["last_input_df"] = input_df
        st.session_state["last_inputs"]   = inputs

    # ── Display result (persists across reruns) ────────────────────────────────
    if "last_result" in st.session_state:
        result   = st.session_state["last_result"]
        input_df = st.session_state["last_input_df"]
        inputs_s = st.session_state["last_inputs"]

        # Risk result card
        render_result(result)

        # Gauge + feature importance
        col_left, col_right = st.columns([1, 1.4])
        with col_left:
            st.markdown("#### 🎯 Risk Gauge")
            fig_gauge = plot_gauge(result.prob_ensemble)
            st.pyplot(fig_gauge, use_container_width=True)
            import matplotlib.pyplot as plt; plt.close(fig_gauge)

        with col_right:
            st.markdown("#### 📊 Feature Importances")
            fig_fi = plot_feature_importance(result.feature_importances)
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)

        st.markdown("<hr class='section'>", unsafe_allow_html=True)

        # What-If Simulator
        with st.expander("🔬 What-If Simulator — adjust risk factors", expanded=False):
            render_whatif(bundle, input_df, result.prob_ensemble)

        st.markdown("<hr class='section'>", unsafe_allow_html=True)

        # PDF download
        st.markdown("#### 📄 Download Report")
        pdf_bytes = generate_pdf_report(inputs_s, result, bundle)
        if pdf_bytes:
            st.download_button(
                label="⬇ Download PDF Clinical Report",
                data=pdf_bytes,
                file_name="cardiorisk_report.pdf",
                mime="application/pdf",
                use_container_width=False,
            )
        else:
            st.info("Install `reportlab` to enable PDF export.")

    # ── EDA / Model Performance ────────────────────────────────────────────────
    st.markdown("<hr class='section'>", unsafe_allow_html=True)
    with st.expander("📈 Model Performance & Dataset Analytics", expanded=False):
        fig_roc = plot_roc_pr(bundle)
        st.pyplot(fig_roc, use_container_width=True)
        import matplotlib.pyplot as plt; plt.close(fig_roc)

        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset Size",   f"{len(df):,} patients")
        c2.metric("Disease Rate",   f"{y.mean()*100:.1f}%")
        c3.metric("Data SHA-256",   sha256, help="First 12 chars of CSV hash")

        # Version footer
        st.markdown(f"""
        <div style='color:#334155; font-size:.8rem; margin-top:12px;'>
        App v{APP_VERSION} · Model v{bundle.version} · Trained {bundle.trained} ·
        Data: {DATA_FILE}
        </div>
        """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("<hr class='section'>", unsafe_allow_html=True)
    render_disclaimer()


if __name__ == "__main__":
    main()
