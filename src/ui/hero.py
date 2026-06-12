"""
Hero banner and favicon injection.
"""
import streamlit as st

from src.config import LOGO_URI


def inject_favicon() -> None:
    """Override Streamlit's default favicon with the CardioRisk logo, if available."""
    if not LOGO_URI:
        return

    st.markdown(f"""
<script>
(function() {{
    const uri = "{LOGO_URI}";
    function setFavicon() {{
        document.querySelectorAll("link[rel*='icon']").forEach(el => el.remove());
        const link = document.createElement('link');
        link.rel = 'icon';
        link.type = 'image/png';
        link.href = uri;
        document.head.appendChild(link);
        const shortcut = document.createElement('link');
        shortcut.rel = 'shortcut icon';
        shortcut.type = 'image/png';
        shortcut.href = uri;
        document.head.appendChild(shortcut);
    }}
    setFavicon();
    setTimeout(setFavicon, 500);
    setTimeout(setFavicon, 1500);
    const obs = new MutationObserver(setFavicon);
    obs.observe(document.head, {{ childList: true, subtree: false }});
}})();
</script>
""", unsafe_allow_html=True)


def render_hero(n_patients: int, xgb_auc: float) -> None:
    st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-chip">Live \u00b7 Predict the Risk</div>
    <h1 class="hero-title">CardioRisk <span>AI</span></h1>
    <p class="hero-sub">
        Ensemble ML system for cardiovascular disease risk stratification \u2014
        combining XGBoost gradient boosting with Logistic Regression on the
        multi-site UCI Heart Disease dataset.
    </p>
    <div class="hero-meta">
        <div class="hero-stat"><span class="hero-stat-val">{n_patients:,}</span><span class="hero-stat-lbl">Patients</span></div>
        <div class="hero-stat"><span class="hero-stat-val">4</span><span class="hero-stat-lbl">Clinical Sites</span></div>
        <div class="hero-stat"><span class="hero-stat-val">13</span><span class="hero-stat-lbl">Features</span></div>
        <div class="hero-stat"><span class="hero-stat-val">2</span><span class="hero-stat-lbl">Models</span></div>
        <div class="hero-stat"><span class="hero-stat-val">{xgb_auc:.3f}</span><span class="hero-stat-lbl">Best AUC</span></div>
    </div>
</div>""", unsafe_allow_html=True)
