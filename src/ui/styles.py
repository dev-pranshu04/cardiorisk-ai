"""
Global CSS injection for the CardioRisk AI dashboard.
"""
import streamlit as st


def inject_global_styles() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #040810 !important;
}
*, *::before, *::after { box-sizing: border-box; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080f1e; }
::-webkit-scrollbar-thumb { background: #1a3a6e; border-radius: 4px; }

/* ── Background mesh ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 80% 50% at 20% 0%, rgba(29,78,216,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(14,165,233,0.08) 0%, transparent 50%),
        radial-gradient(ellipse 40% 60% at 50% 50%, rgba(6,182,212,0.04) 0%, transparent 70%);
}
.stApp > * { position: relative; z-index: 1; }

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #06080f 0%, #0c1628 40%, #0e1e3f 100%);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 24px;
    padding: 44px 52px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: 0; right: 0; bottom: 0;
    width: 40%;
    background: radial-gradient(ellipse at 80% 50%, rgba(37,99,235,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute; bottom: -1px; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.4), transparent);
}
.hero-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(37,99,235,0.12);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 10px; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; color: #60a5fa;
    margin-bottom: 16px;
}
.hero-chip::before { content: '●'; font-size: 7px; color: #22c55e; }
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 42px; font-weight: 800;
    color: #f1f5f9; line-height: 1.1;
    letter-spacing: -0.5px; margin: 0 0 10px;
}
.hero-title span { color: #3b82f6; }
.hero-sub {
    font-size: 14px; color: rgba(255,255,255,0.38);
    font-weight: 400; margin: 0; line-height: 1.6;
    max-width: 560px;
}
.hero-meta {
    display: flex; gap: 24px; margin-top: 24px; flex-wrap: wrap;
}
.hero-stat {
    display: flex; flex-direction: column; gap: 2px;
}
.hero-stat-val {
    font-family: 'DM Mono', monospace;
    font-size: 20px; font-weight: 500; color: #e2e8f0;
}
.hero-stat-lbl {
    font-size: 10px; color: #334155; letter-spacing: 1px;
    text-transform: uppercase; font-weight: 500;
}

/* ── KPI Strip ── */
.kpi-row {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 36px;
}
.kpi-card {
    background: #070d1c;
    border: 1px solid #0f1f3a;
    border-radius: 16px;
    padding: 22px 24px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(59,130,246,0.3); }
.kpi-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, linear-gradient(90deg,#1d4ed8,#3b82f6));
}
.kpi-val {
    font-family: 'DM Mono', monospace;
    font-size: 32px; font-weight: 500; color: #60a5fa; line-height: 1;
}
.kpi-lbl {
    font-size: 10px; color: #334155; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; margin-top: 8px;
}
.kpi-delta {
    font-size: 11px; color: #22c55e; margin-top: 4px; font-family: 'DM Mono', monospace;
}

/* ── Section Label ── */
.sec-hd {
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #1d4ed8;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(29,78,216,0.15);
    margin-bottom: 18px;
}

/* ── Model Cards ── */
.model-card {
    background: #070d1c; border: 1px solid #0f1f3a;
    border-radius: 14px; padding: 20px 22px; margin-bottom: 14px;
    transition: transform 0.15s, border-color 0.15s;
}
.model-card:hover { transform: translateY(-1px); border-color: #1a3a6e; }
.model-card.primary {
    border-color: #1d4ed8;
    background: linear-gradient(135deg, #060e24, #0c1c40);
}
.model-tag {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 8px;
}
.model-name { font-size: 11px; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 1px; }
.model-badge {
    background: #1d4ed8; color: #bfdbfe;
    font-size: 8px; font-weight: 700; padding: 2px 8px;
    border-radius: 4px; letter-spacing: 0.5px; text-transform: uppercase;
}
.model-auc {
    font-family: 'DM Mono', monospace;
    font-size: 34px; font-weight: 500; color: #f1f5f9; line-height: 1;
}
.model-meta { font-size: 11px; color: #1e3a5f; margin-top: 6px; font-family: 'DM Mono', monospace; }

/* ── Info Panel ── */
.info-panel {
    background: #070d1c; border: 1px solid #0f1f3a;
    border-radius: 14px; padding: 22px;
}
.info-panel p { font-size: 13px; color: #475569; line-height: 1.75; margin: 0; }
.info-panel strong { color: #64748b; font-weight: 500; }
.info-panel .warning {
    display: flex; align-items: flex-start; gap: 10px;
    background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.15);
    border-radius: 8px; padding: 12px 14px; margin-top: 14px;
    font-size: 11px; color: #78716c; line-height: 1.6;
}
.info-panel .warning::before { content: '⚠'; font-size: 14px; flex-shrink: 0; color: #f59e0b; }

/* ── Risk Output ── */
.risk-display {
    border-radius: 20px; padding: 32px 36px;
    text-align: center; margin-bottom: 24px; position: relative; overflow: hidden;
}
.risk-display.low  {
    background: linear-gradient(135deg,#031409,#042d1a,#053a21);
    border: 1px solid rgba(5,150,105,0.35);
}
.risk-display.med  {
    background: linear-gradient(135deg,#1a0a00,#2d1a00,#3d2000);
    border: 1px solid rgba(217,119,6,0.35);
}
.risk-display.high {
    background: linear-gradient(135deg,#1a0000,#2d0505,#3d0a0a);
    border: 1px solid rgba(220,38,38,0.35);
}
.risk-pct {
    font-family: 'Playfair Display', serif;
    font-size: 64px; font-weight: 800; color: #fff; line-height: 1;
}
.risk-label {
    font-size: 12px; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: rgba(255,255,255,0.55);
    margin-top: 10px;
}
.risk-note {
    font-size: 12px; color: rgba(255,255,255,0.35);
    margin-top: 14px; line-height: 1.7; max-width: 280px; margin-left: auto; margin-right: auto;
}

/* ── Progress Bars ── */
.pbar-wrap { margin: 12px 0; }
.pbar-header {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 11px; color: #334155; margin-bottom: 6px;
}
.pbar-header .val { font-family: 'DM Mono', monospace; color: #60a5fa; font-weight: 500; }
.pbar-track { background: #060c1a; border-radius: 3px; height: 4px; overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }

/* ── Patient Table ── */
.ptable-wrap { overflow: hidden; border-radius: 12px; border: 1px solid #0f1f3a; }
.ptable { width: 100%; border-collapse: collapse; font-size: 12px; }
.ptable thead th {
    background: #04080f; color: #1e3a5f;
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; padding: 10px 14px; text-align: left;
}
.ptable tbody td { padding: 8px 14px; border-bottom: 1px solid #080f1a; }
.ptable tbody td:first-child { color: #334155; font-size: 11px; }
.ptable tbody td:last-child {
    color: #cbd5e1; font-weight: 500; font-family: 'DM Mono', monospace; font-size: 11px;
}
.ptable tbody tr:last-child td { border-bottom: none; }
.ptable tbody tr:hover td { background: #070d1c; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #04080e !important;
    border-right: 1px solid #0a1428 !important;
}
[data-testid="stSidebarContent"] {
    padding-bottom: 60px !important;
}
[data-testid="stSidebar"] label {
    color: #334155 !important; font-size: 10px !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.8px !important; font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #1d4ed8 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #070d1c !important;
    border: 1px solid #0f1f3a !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
}

/* ── Sidebar compact spacing ── */
[data-testid="stSidebar"] .stSlider { margin-bottom: 4px !important; padding-bottom: 0 !important; }
[data-testid="stSidebar"] .stSelectbox { margin-bottom: 4px !important; }
[data-testid="stSidebar"] .stRadio { margin-bottom: 4px !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 6px !important; }
[data-testid="stSidebar"] .element-container { margin-bottom: 4px !important; }
[data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0 !important; }
[data-testid="stSidebar"] hr { margin: 10px 0 !important; }
.stButton > button {
    background: linear-gradient(135deg,#1d4ed8 0%,#2563eb 50%,#3b82f6 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; padding: 14px !important;
    font-weight: 600 !important; font-size: 13px !important;
    width: 100% !important; letter-spacing: 0.5px !important;
    box-shadow: 0 4px 24px rgba(37,99,235,0.4) !important;
    transition: all 0.2s !important; font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 32px rgba(37,99,235,0.6) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #0a1428 !important; margin: 28px 0 !important; }

/* ── Matplotlib figs ── */
.stImage img { border-radius: 12px; }

/* ── Tab-like metric chips ── */
.metric-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
.metric-chip {
    background: #070d1c; border: 1px solid #0f1f3a;
    border-radius: 8px; padding: 6px 14px;
    font-family: 'DM Mono', monospace; font-size: 11px; color: #475569;
}
.metric-chip strong { color: #60a5fa; font-weight: 500; }
</style>
""", unsafe_allow_html=True)
