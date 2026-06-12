"""
SHAP-based per-patient explainability (P1-6).

Adds a waterfall chart showing how each feature pushed this patient's
XGBoost prediction above/below the model's expected (base) value.
"""
import matplotlib.pyplot as plt
import streamlit as st

from src.config import BG2, T1

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHAP_AVAILABLE = False


@st.cache_resource
def get_explainer(_xgb_model):
    """Cache the SHAP TreeExplainer for the trained XGBoost model."""
    return shap.TreeExplainer(_xgb_model)


def render_shap_waterfall(xgb_model, Xi_imp, flabels, feats):
    """Render a per-patient SHAP waterfall chart for the XGBoost prediction.

    Falls back to a friendly message if the `shap` package isn't installed.
    """
    if not SHAP_AVAILABLE:
        st.info(
            "SHAP explanations are unavailable because the `shap` package "
            "is not installed. Add `shap` to requirements.txt to enable "
            "per-patient feature attribution."
        )
        return

    explainer = get_explainer(xgb_model)
    sv = explainer(Xi_imp)
    sv.feature_names = [flabels.get(f, f) for f in feats]

    fig = plt.figure(figsize=(8, 5), facecolor=BG2)
    try:
        shap.plots.waterfall(sv[0], show=False, max_display=13)
        fig = plt.gcf()
        fig.set_facecolor(BG2)
        for a in fig.axes:
            a.set_facecolor(BG2)
            a.tick_params(colors=T1)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)
