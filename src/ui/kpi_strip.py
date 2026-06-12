"""
Top-of-page KPI strip.
"""
import streamlit as st


def render_kpi_strip(xgb_auc, cv_auc, lr_auc, prevalence, n_positive) -> None:
    st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card" style="--accent:linear-gradient(90deg,#1d4ed8,#3b82f6);">
        <div class="kpi-val">{xgb_auc:.3f}</div>
        <div class="kpi-lbl">XGBoost AUC-ROC</div>
        <div class="kpi-delta">\u2191 Primary Model</div>
    </div>
    <div class="kpi-card" style="--accent:linear-gradient(90deg,#0369a1,#06b6d4);">
        <div class="kpi-val">{cv_auc:.3f}</div>
        <div class="kpi-lbl">5-Fold CV AUC</div>
        <div class="kpi-delta">Stratified \u00b7 Robust</div>
    </div>
    <div class="kpi-card" style="--accent:linear-gradient(90deg,#6d28d9,#8b5cf6);">
        <div class="kpi-val">{lr_auc:.3f}</div>
        <div class="kpi-lbl">LR AUC-ROC</div>
        <div class="kpi-delta">Interpretable Baseline</div>
    </div>
    <div class="kpi-card" style="--accent:linear-gradient(90deg,#047857,#10b981);">
        <div class="kpi-val">{prevalence*100:.0f}%</div>
        <div class="kpi-lbl">Disease Prevalence</div>
        <div class="kpi-delta">{int(n_positive)} positive cases</div>
    </div>
</div>""", unsafe_allow_html=True)
