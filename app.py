import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI · Clinical Decision Support",
    page_icon="assets/Cardiorisk_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
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
    overflow-y: auto !important;
    overflow-x: hidden !important;
}
[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
    height: 100vh !important;
    padding-bottom: 60px !important;
}
[data-testid="stSidebarContent"] {
    overflow-y: auto !important;
    height: 100% !important;
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

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
BG   = "#040810"
BG2  = "#070d1c"
GRID = "#0a1428"
T0   = "#334155"
T1   = "#475569"
T2   = "#64748b"
T3   = "#94a3b8"
BLUE = "#3b82f6"
RED  = "#ef4444"
GRN  = "#10b981"
PRP  = "#8b5cf6"
CYN  = "#06b6d4"
ORG  = "#f59e0b"

plt.rcParams.update({
    "figure.facecolor": BG2, "axes.facecolor": BG2,
    "axes.edgecolor": GRID, "axes.labelcolor": T1,
    "xtick.color": T0, "ytick.color": T0,
    "text.color": T2, "grid.color": GRID, "grid.linewidth": 0.4,
    "font.family": "DejaVu Sans", "font.size": 9,
})

FEATS = ["age","sex_enc","cp_enc","trestbps","chol","fbs_enc","restecg_enc",
         "thalch","exang_enc","oldpeak","slope_enc","ca","thal_enc"]

FLABELS = {
    "age":        "Age",
    "sex_enc":    "Biological Sex",
    "cp_enc":     "Chest Pain Type",
    "trestbps":   "Resting BP",
    "chol":       "Cholesterol",
    "fbs_enc":    "Fasting Blood Sugar",
    "restecg_enc":"Resting ECG",
    "thalch":     "Max Heart Rate",
    "exang_enc":  "Exercise Angina",
    "oldpeak":    "ST Depression",
    "slope_enc":  "ST Slope",
    "ca":         "Major Vessels",
    "thal_enc":   "Thalassemia"
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_uci.csv")
    df["target"]      = (df["num"] > 0).astype(int)
    df["sex_enc"]     = (df["sex"] == "Male").astype(int)
    df["cp_enc"]      = df["cp"].map({"typical angina":0,"atypical angina":1,"non-anginal":2,"asymptomatic":3})
    df["fbs_enc"]     = df["fbs"].map({True:1,False:0,"True":1,"False":0})
    df["restecg_enc"] = df["restecg"].map({"normal":0,"st-t abnormality":1,"lv hypertrophy":2})
    df["exang_enc"]   = df["exang"].map({True:1,False:0,"True":1,"False:0":0})
    df["slope_enc"]   = df["slope"].map({"upsloping":0,"flat":1,"downsloping":2})
    df["thal_enc"]    = df["thal"].map({"normal":1,"fixed defect":2,"reversable defect":3})
    X = df[FEATS]
    y = df["target"]
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=FEATS)
    return df, X_imp, y, imp

@st.cache_resource
def train_models(_X, _y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=0.5, solver="lbfgs", random_state=42)
    lr.fit(X_tr_s, y_tr)
    lr_prob = lr.predict_proba(X_te_s)[:,1]
    lr_auc  = roc_auc_score(y_te, lr_prob)
    lr_ap   = average_precision_score(y_te, lr_prob)

    # XGBoost — 150 trees is sufficient for this dataset size; keeps cold-start fast
    sp = (y_tr == 0).sum() / (y_tr == 1).sum()
    xgb_m = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=sp, eval_metric="logloss",
        random_state=42, verbosity=0, device="cpu",
        n_jobs=-1
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    xgb_prob = xgb_m.predict_proba(X_te)[:,1]
    xgb_auc  = roc_auc_score(y_te, xgb_prob)
    xgb_ap   = average_precision_score(y_te, xgb_prob)

    # Single CV call with multiple metrics to avoid running 5-fold twice
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(xgb_m, _X, _y, cv=skf, scoring="roc_auc", n_jobs=-1).mean()
    cv_ap  = cross_val_score(xgb_m, _X, _y, cv=skf, scoring="average_precision", n_jobs=-1).mean()

    return lr, xgb_m, sc, lr_auc, xgb_auc, cv_auc, lr_ap, xgb_ap, cv_ap, X_te, y_te, X_te_s

df_raw, X, y, imputer = load_data()

# Show a spinner on cold start (cache miss); subsequent loads are instant
if "models_loaded" not in st.session_state:
    with st.spinner("🫀 Initialising CardioRisk AI — training models on first load (~10s)..."):
        lr_m, xgb_m, scaler, lr_auc, xgb_auc, cv_auc, lr_ap, xgb_ap, cv_ap, Xte, yte, Xte_s = train_models(X, y)
    st.session_state["models_loaded"] = True
else:
    lr_m, xgb_m, scaler, lr_auc, xgb_auc, cv_auc, lr_ap, xgb_ap, cv_ap, Xte, yte, Xte_s = train_models(X, y)
prevalence = y.mean()
n_patients = len(df_raw)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 16px;'>
        <div style='margin-bottom:6px;'>
    <img src='assets/Cardiorisk_logo.png' width='60'>
</div>
        <div style='font-family:"Playfair Display",serif;font-size:16px;font-weight:700;color:#e2e8f0;'>CardioRisk AI</div>
        <div style='font-size:9px;color:#1e3a5f;margin-top:3px;letter-spacing:2px;text-transform:uppercase;'>Clinical Input Panel</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="color:#1d4ed8;font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:8px 0 6px;">Demographics</p>', unsafe_allow_html=True)
    age     = st.slider("Age (years)", int(df_raw.age.min()), int(df_raw.age.max()), 54)
    sex     = st.selectbox("Biological Sex", ["Male","Female"])

    st.markdown('<p style="color:#1d4ed8;font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:8px 0 6px;">Symptoms</p>', unsafe_allow_html=True)
    cp      = st.selectbox("Chest Pain Type", ["Asymptomatic","Typical Angina","Atypical Angina","Non-Anginal"])
    exang   = st.radio("Exercise-Induced Angina", ["No","Yes"], horizontal=True)

    st.markdown('<p style="color:#1d4ed8;font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:8px 0 6px;">Vitals & Labs</p>', unsafe_allow_html=True)
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
    chol     = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 240)
    fbs      = st.radio("Fasting Blood Sugar > 120 mg/dL", ["No","Yes"], horizontal=True)
    thalch   = st.slider("Maximum Heart Rate (bpm)", 70, 210, 150)

    st.markdown('<p style="color:#1d4ed8;font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:8px 0 6px;">ECG & Imaging</p>', unsafe_allow_html=True)
    restecg  = st.selectbox("Resting ECG Result", ["Normal","ST-T Abnormality","LV Hypertrophy"])
    oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope    = st.selectbox("ST Slope", ["Upsloping","Flat","Downsloping"])
    ca       = st.slider("Fluoroscopy Vessels (0–3)", 0, 3, 0)
    thal     = st.selectbox("Thalassemia", ["Normal","Fixed Defect","Reversable Defect"])

    st.markdown("<br>", unsafe_allow_html=True)
    go = st.button("🔍  Compute Cardiac Risk Score")

# ─────────────────────────────────────────────────────────────────────────────
# ENCODING
# ─────────────────────────────────────────────────────────────────────────────
def encode_input():
    return pd.DataFrame([[
        age,
        1 if sex == "Male" else 0,
        {"Typical Angina":0,"Atypical Angina":1,"Non-Anginal":2,"Asymptomatic":3}[cp],
        trestbps, chol,
        1 if fbs == "Yes" else 0,
        {"Normal":0,"ST-T Abnormality":1,"LV Hypertrophy":2}[restecg],
        thalch,
        1 if exang == "Yes" else 0,
        oldpeak,
        {"Upsloping":0,"Flat":1,"Downsloping":2}[slope],
        ca,
        {"Normal":1,"Fixed Defect":2,"Reversable Defect":3}[thal]
    ]], columns=FEATS)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-chip">Live · Inference Ready</div>
    <h1 class="hero-title">CardioRisk <span>AI</span></h1>
    <p class="hero-sub">
        Ensemble ML system for cardiovascular disease risk stratification —
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

# ─────────────────────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card" style="--accent:linear-gradient(90deg,#1d4ed8,#3b82f6);">
        <div class="kpi-val">{xgb_auc:.3f}</div>
        <div class="kpi-lbl">XGBoost AUC-ROC</div>
        <div class="kpi-delta">↑ Primary Model</div>
    </div>
    <div class="kpi-card" style="--accent:linear-gradient(90deg,#0369a1,#06b6d4);">
        <div class="kpi-val">{cv_auc:.3f}</div>
        <div class="kpi-lbl">5-Fold CV AUC</div>
        <div class="kpi-delta">Stratified · Robust</div>
    </div>
    <div class="kpi-card" style="--accent:linear-gradient(90deg,#6d28d9,#8b5cf6);">
        <div class="kpi-val">{lr_auc:.3f}</div>
        <div class="kpi-lbl">LR AUC-ROC</div>
        <div class="kpi-delta">Interpretable Baseline</div>
    </div>
    <div class="kpi-card" style="--accent:linear-gradient(90deg,#047857,#10b981);">
        <div class="kpi-val">{prevalence*100:.0f}%</div>
        <div class="kpi-lbl">Disease Prevalence</div>
        <div class="kpi-delta">{int(y.sum())} positive cases</div>
    </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD ROW
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1, 1.75, 1])

with c1:
    st.markdown('<p class="sec-hd">Model Registry</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-card primary">
        <div class="model-tag">
            <span class="model-name">XGBoost v2</span>
            <span class="model-badge">PRIMARY · 65%</span>
        </div>
        <div class="model-auc">{xgb_auc:.3f}</div>
        <div class="model-meta">AUC · CV={cv_auc:.3f} · AP={xgb_ap:.3f}</div>
    </div>
    <div class="model-card">
        <div class="model-tag">
            <span class="model-name">Logistic Regression</span>
            <span style="font-size:9px;color:#1e3a5f;">35%</span>
        </div>
        <div class="model-auc" style="font-size:26px;">{lr_auc:.3f}</div>
        <div class="model-meta">AUC · AP={lr_ap:.3f} · Interpretable</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-hd" style="margin-top:20px;">Ensemble Strategy</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-chips">
        <span class="metric-chip">XGB <strong>65%</strong></span>
        <span class="metric-chip">LR <strong>35%</strong></span>
        <span class="metric-chip">Weighted avg</span>
    </div>
    <p style="font-size:11px;color:#1e3a5f;line-height:1.7;">
        Soft-probability blending combines XGBoost's
        non-linear feature interactions with Logistic Regression's
        calibrated linear baseline for robust generalization.
    </p>""", unsafe_allow_html=True)

with c2:
    st.markdown('<p class="sec-hd">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    fig = plt.figure(figsize=(10, 4.2), facecolor=BG2)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    ax0 = fig.add_subplot(gs[0])
    vals = y.value_counts().sort_index()
    colors_pie = [BLUE, RED]
    wedges, texts, autotexts = ax0.pie(
        [vals[0], vals[1]], labels=["Healthy","Disease"],
        colors=colors_pie, autopct="%1.0f%%", startangle=90,
        textprops={"fontsize":8, "color":T2},
        wedgeprops={"linewidth":2.5, "edgecolor":BG2},
        pctdistance=0.75
    )
    for at in autotexts: at.set_color("#f1f5f9"); at.set_fontweight("600")
    ax0.set_title("Class Distribution", fontsize=9, fontweight="600", color=T1, pad=10)

    ax1 = fig.add_subplot(gs[1])
    ax1.hist(df_raw[df_raw.target==0]["age"], bins=20, alpha=0.8, color=BLUE, label="Healthy", linewidth=0, rwidth=0.9)
    ax1.hist(df_raw[df_raw.target==1]["age"], bins=20, alpha=0.75, color=RED, label="Disease", linewidth=0, rwidth=0.9)
    ax1.set_title("Age by Outcome", fontsize=9, fontweight="600", color=T1, pad=10)
    ax1.set_xlabel("Age (years)", fontsize=8)
    ax1.set_ylabel("Count", fontsize=8)
    ax1.legend(fontsize=7.5, framealpha=0, labelcolor=T2)
    ax1.grid(axis="y", alpha=0.35)
    for sp in ["top","right"]: ax1.spines[sp].set_visible(False)

    ax2 = fig.add_subplot(gs[2])
    src = df_raw["dataset"].value_counts()
    bar_colors = [BLUE, GRN, ORG, PRP][:len(src)]
    bars = ax2.bar(src.index, src.values, color=bar_colors, edgecolor=BG2, linewidth=2, width=0.55)
    ax2.set_title("Patients by Site", fontsize=9, fontweight="600", color=T1, pad=10)
    ax2.tick_params(axis="x", rotation=18, labelsize=7)
    ax2.grid(axis="y", alpha=0.35)
    for b in bars:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+3,
                 str(int(b.get_height())), ha="center", fontsize=8, color=T1)
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)

    plt.tight_layout(pad=1.2)
    st.pyplot(fig, width="stretch")
    plt.close()

with c3:
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
            Research & educational use only. Not a certified medical device.
            Do not use for clinical diagnosis without physician oversight.
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION BLOCK
# ─────────────────────────────────────────────────────────────────────────────
if go:
    Xi     = encode_input()
    Xi_imp = pd.DataFrame(imputer.transform(Xi), columns=FEATS)
    xp     = xgb_m.predict_proba(Xi_imp)[0][1]
    lp     = lr_m.predict_proba(scaler.transform(Xi_imp))[0][1]
    ep     = xp * 0.65 + lp * 0.35

    risk_cls  = "low" if ep < 0.30 else "med" if ep < 0.60 else "high"
    risk_tag  = "LOW RISK" if ep < 0.30 else "MODERATE RISK" if ep < 0.60 else "HIGH RISK"
    risk_note = (
        "Routine monitoring advised. Continue healthy lifestyle habits and annual review."
        if ep < 0.30 else
        "Further evaluation recommended — consider stress test, echocardiogram, and full lipid panel."
        if ep < 0.60 else
        "Cardiology referral strongly indicated. Prompt clinical evaluation required."
    )

    # ── Row 1: Risk + breakdown + features ─────────────────────────────────
    r1, r2 = st.columns([1, 1.9])

    with r1:
        st.markdown('<p class="sec-hd">Risk Score</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-display {risk_cls}">
            <div class="risk-pct">{ep*100:.1f}%</div>
            <div class="risk-label">{risk_tag}</div>
            <div class="risk-note">{risk_note}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<p class="sec-hd" style="margin-top:4px;">Model Contributions</p>', unsafe_allow_html=True)
        for nm, prob, col in [
            ("XGBoost (65%)",           xp, BLUE),
            ("Logistic Regression (35%)", lp, PRP),
            ("Ensemble Output",          ep, GRN),
        ]:
            st.markdown(f"""
            <div class="pbar-wrap">
                <div class="pbar-header">
                    <span>{nm}</span>
                    <span class="val">{prob*100:.1f}%</span>
                </div>
                <div class="pbar-track">
                    <div class="pbar-fill" style="width:{prob*100:.1f}%;background:{col};"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown('<p class="sec-hd">Feature Importance — XGBoost (Gain)</p>', unsafe_allow_html=True)
        imp_vals = xgb_m.feature_importances_
        fi = pd.DataFrame({
            "feature": [FLABELS[f] for f in FEATS],
            "importance": imp_vals
        }).sort_values("importance", ascending=True)

        fig2, ax = plt.subplots(figsize=(8, 5), facecolor=BG2)
        bar_cols = [RED if v > 0.09 else BLUE for v in fi["importance"]]
        bars = ax.barh(fi["feature"], fi["importance"],
                       color=bar_cols, edgecolor=BG, linewidth=1, height=0.65)
        ax.axvline(0.09, color=GRID, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Feature Importance (Gain)", fontsize=9, color=T1)
        ax.grid(axis="x", alpha=0.25)
        for b, val in zip(bars, fi["importance"]):
            ax.text(val + 0.003, b.get_y() + b.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8, color=T0)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        p1 = mpatches.Patch(color=RED, label=">9% — High signal")
        p2 = mpatches.Patch(color=BLUE, label="Standard signal")
        ax.legend(handles=[p1, p2], fontsize=8, framealpha=0.05,
                  loc="lower right", labelcolor=T2)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig2, width="stretch")
        plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 2: ROC + PR + Calibration + CM ─────────────────────────────────
    st.markdown('<p class="sec-hd">Model Diagnostics</p>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)

    xgb_prob_te = xgb_m.predict_proba(Xte)[:,1]
    lr_prob_te  = lr_m.predict_proba(Xte_s)[:,1]

    # ROC Curve
    with d1:
        fig3, ax = plt.subplots(figsize=(4, 3.6), facecolor=BG2)
        for prob, lbl, col in [(xgb_prob_te,"XGBoost",BLUE),(lr_prob_te,"LR",PRP)]:
            fpr, tpr, _ = roc_curve(yte, prob)
            auc = roc_auc_score(yte, prob)
            ax.plot(fpr, tpr, color=col, lw=1.6, label=f"{lbl} ({auc:.3f})")
        ax.plot([0,1],[0,1], color=GRID, lw=0.8, linestyle="--")
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.set_title("ROC Curve", fontsize=9, fontweight="600", color=T1, pad=8)
        ax.legend(fontsize=7.5, framealpha=0.05, labelcolor=T2)
        ax.grid(alpha=0.2); ax.set_xlim(0,1); ax.set_ylim(0,1)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig3, width="stretch")
        plt.close()

    # PR Curve
    with d2:
        fig4, ax = plt.subplots(figsize=(4, 3.6), facecolor=BG2)
        for prob, lbl, col in [(xgb_prob_te,"XGBoost",BLUE),(lr_prob_te,"LR",PRP)]:
            p_, r_, _ = precision_recall_curve(yte, prob)
            ap = average_precision_score(yte, prob)
            ax.plot(r_, p_, color=col, lw=1.6, label=f"{lbl} AP={ap:.3f}")
        ax.axhline(yte.mean(), color=GRID, linestyle="--", lw=0.8)
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.set_title("Precision–Recall", fontsize=9, fontweight="600", color=T1, pad=8)
        ax.legend(fontsize=7.5, framealpha=0.05, labelcolor=T2)
        ax.grid(alpha=0.2); ax.set_xlim(0,1); ax.set_ylim(0,1)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig4, width="stretch")
        plt.close()

    # Calibration Plot
    with d3:
        fig5, ax = plt.subplots(figsize=(4, 3.6), facecolor=BG2)
        for prob, lbl, col in [(xgb_prob_te,"XGBoost",BLUE),(lr_prob_te,"LR",PRP)]:
            frac_pos, mean_pred = calibration_curve(yte, prob, n_bins=8)
            ax.plot(mean_pred, frac_pos, marker="o", ms=4, color=col, lw=1.5, label=lbl)
        ax.plot([0,1],[0,1], color=GRID, linestyle="--", lw=0.8, label="Perfect")
        ax.set_xlabel("Mean Predicted Probability", fontsize=8)
        ax.set_ylabel("Fraction of Positives", fontsize=8)
        ax.set_title("Calibration Curve", fontsize=9, fontweight="600", color=T1, pad=8)
        ax.legend(fontsize=7.5, framealpha=0.05, labelcolor=T2)
        ax.grid(alpha=0.2); ax.set_xlim(0,1); ax.set_ylim(0,1)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig5, width="stretch")
        plt.close()

    # Confusion Matrix (XGBoost)
    with d4:
        fig6, ax = plt.subplots(figsize=(4, 3.6), facecolor=BG2)
        cm = confusion_matrix(yte, xgb_m.predict(Xte))
        cmap = LinearSegmentedColormap.from_list("blues", [BG2, BLUE], N=256)
        im = ax.imshow(cm, cmap=cmap, vmin=0)
        ax.set_title("XGBoost Confusion Matrix", fontsize=9, fontweight="600", color=T1, pad=8)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Predicted Neg","Predicted Pos"], fontsize=7.5)
        ax.set_yticklabels(["Actual Neg","Actual Pos"], fontsize=7.5)
        thresh = cm.max() * 0.55
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r,c]), ha="center", va="center",
                        fontsize=18, fontweight="700",
                        color="#fff" if cm[r,c] > thresh else T1)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig6, width="stretch")
        plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 3: Classification Report + Patient Table ────────────────────────
    p1, p2 = st.columns([1.1, 1])

    with p1:
        st.markdown('<p class="sec-hd">Classification Report — XGBoost</p>', unsafe_allow_html=True)
        cr = classification_report(yte, xgb_m.predict(Xte), output_dict=True)
        rows_html = ""
        for cls_name, label in [("0","Healthy (0)"),("1","Disease (1)"),("macro avg","Macro Avg"),("weighted avg","Weighted Avg")]:
            if cls_name in cr:
                d = cr[cls_name]
                supp = f"{int(d.get('support',0))}" if 'support' in d else "—"
                rows_html += f"""
                <tr>
                    <td>{label}</td>
                    <td>{d['precision']:.3f}</td>
                    <td>{d['recall']:.3f}</td>
                    <td>{d['f1-score']:.3f}</td>
                    <td>{supp}</td>
                </tr>"""
        st.markdown(f"""
        <div class="ptable-wrap">
        <table class="ptable">
            <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>""", unsafe_allow_html=True)

        # Threshold analysis
        st.markdown('<p class="sec-hd" style="margin-top:18px;">Threshold Sensitivity (XGBoost)</p>', unsafe_allow_html=True)
        fig7, ax = plt.subplots(figsize=(6.5, 2.5), facecolor=BG2)
        thresholds = np.linspace(0.1, 0.9, 80)
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            preds = (xgb_prob_te >= t).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            precisions.append(precision_score(yte, preds, zero_division=0))
            recalls.append(recall_score(yte, preds, zero_division=0))
            f1s.append(f1_score(yte, preds, zero_division=0))
        ax.plot(thresholds, precisions, color=BLUE, lw=1.5, label="Precision")
        ax.plot(thresholds, recalls, color=RED, lw=1.5, label="Recall")
        ax.plot(thresholds, f1s, color=GRN, lw=1.8, label="F1-Score", linestyle="--")
        ax.axvline(0.5, color=T0, linestyle=":", lw=0.8)
        ax.set_xlabel("Decision Threshold", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.legend(fontsize=7.5, framealpha=0.05, labelcolor=T2, ncol=3)
        ax.grid(alpha=0.2); ax.set_xlim(0.1, 0.9); ax.set_ylim(0, 1.05)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(pad=0.8)
        st.pyplot(fig7, width="stretch")
        plt.close()

    with p2:
        st.markdown('<p class="sec-hd">Patient Input Summary</p>', unsafe_allow_html=True)
        patient_rows = [
            ("Age", f"{age} years"), ("Sex", sex), ("Chest Pain", cp),
            ("Resting BP", f"{trestbps} mmHg"), ("Cholesterol", f"{chol} mg/dL"),
            ("Fasting Blood Sugar", fbs), ("Resting ECG", restecg),
            ("Max Heart Rate", f"{thalch} bpm"), ("Exercise Angina", exang),
            ("ST Depression", f"{oldpeak}"), ("ST Slope", slope),
            ("Fluoroscopy Vessels", str(ca)), ("Thalassemia", thal),
        ]
        tbl = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in patient_rows
        )
        st.markdown(f"""
        <div class="ptable-wrap">
        <table class="ptable">
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>{tbl}</tbody>
        </table>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXPLORATORY VIEW (no prediction yet)
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown('<p class="sec-hd">Exploratory Analysis</p>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)

    with e1:
        fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG2)
        cp_g = df_raw.groupby(["cp","target"]).size().unstack(fill_value=0)
        x_pos = np.arange(len(cp_g)); w = 0.38
        if 0 in cp_g.columns: ax.bar(x_pos - w/2, cp_g[0], width=w, color=BLUE, edgecolor=BG, linewidth=1.5, label="Healthy", alpha=0.9)
        if 1 in cp_g.columns: ax.bar(x_pos + w/2, cp_g[1], width=w, color=RED,  edgecolor=BG, linewidth=1.5, label="Disease", alpha=0.9)
        ax.set_xticks(x_pos); ax.set_xticklabels(cp_g.index, rotation=14, fontsize=8)
        ax.set_title("Chest Pain Type vs. Disease Outcome", fontsize=10, fontweight="600", color=T1, pad=10)
        ax.set_ylabel("Patient Count", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.05, labelcolor=T2)
        ax.grid(axis="y", alpha=0.3)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(pad=1.2); st.pyplot(fig, width="stretch"); plt.close()

    with e2:
        fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG2)
        ax.hist(df_raw[df_raw.target==0]["thalch"].dropna(), bins=24, alpha=0.8, color=BLUE, label="Healthy", linewidth=0, rwidth=0.88)
        ax.hist(df_raw[df_raw.target==1]["thalch"].dropna(), bins=24, alpha=0.75, color=RED, label="Disease", linewidth=0, rwidth=0.88)
        ax.set_title("Maximum Heart Rate vs. Disease Outcome", fontsize=10, fontweight="600", color=T1, pad=10)
        ax.set_xlabel("Max Heart Rate (bpm)", fontsize=9)
        ax.set_ylabel("Patient Count", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.05, labelcolor=T2)
        ax.grid(axis="y", alpha=0.3)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(pad=1.2); st.pyplot(fig, width="stretch"); plt.close()

    st.markdown("""
    <div style='text-align:center;padding:28px 0 12px;font-size:13px;color:#1e3a5f;'>
        ← &nbsp; Configure patient vitals and click
        <span style='color:#3b82f6;font-weight:600;'>Compute Cardiac Risk Score</span>
        to view the full diagnostic report
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin-top:40px;'>", unsafe_allow_html=True)
st.markdown("""
<div style="max-width:580px;margin:0 auto;
    background:linear-gradient(135deg,#04080e,#070d1c);
    border:1px solid #0a1428;border-radius:20px;padding:28px 36px;text-align:center;">
    <div style="font-size:9px;font-weight:700;letter-spacing:2.5px;
        text-transform:uppercase;color:#1d4ed8;margin-bottom:8px;">Crafted by</div>
    <div style="font-size:22px;font-weight:700;color:#e2e8f0;letter-spacing:-0.3px;margin-bottom:3px;">Pranshu Kumar</div>
    <div style="font-size:11px;color:#1e3a5f;margin-bottom:22px;">ML Engineer &nbsp;·&nbsp; Healthcare AI &nbsp;·&nbsp; Python</div>
    <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">
        <a href="https://github.com/dev-pranshu04" target="_blank"
           style="display:inline-flex;align-items:center;gap:7px;padding:9px 20px;
           border-radius:9px;font-size:12px;font-weight:500;text-decoration:none;
           background:#0a0f18;border:1px solid #141e30;color:#94a3b8;">
           GitHub
        </a>
        <a href="https://www.linkedin.com/in/dev-pranshu" target="_blank"
           style="display:inline-flex;align-items:center;gap:7px;padding:9px 20px;
           border-radius:9px;font-size:12px;font-weight:500;text-decoration:none;
           background:#0a66c2;color:#fff;">
           LinkedIn
        </a>
        <a href="https://www.instagram.com/im_pranshu29/" target="_blank"
           style="display:inline-flex;align-items:center;gap:7px;padding:9px 20px;
           border-radius:9px;font-size:12px;font-weight:500;text-decoration:none;
           background:linear-gradient(135deg,#833ab4,#e1306c,#f77737);color:#fff;">
           Instagram
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
