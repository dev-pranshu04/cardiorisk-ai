import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: dark-mode safe, industry-grade ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    color: white;
}
.hero-banner h1 {
    font-size: 28px;
    font-weight: 800;
    color: white !important;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    font-size: 14px;
    color: rgba(255,255,255,0.75);
    margin: 0;
}
.hero-badges {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.hero-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
    font-size: 11px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}

/* ── KPI cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.kpi-card {
    background: #1e3a8a;
    border: 1px solid #2563eb;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #60a5fa;
    line-height: 1.1;
}
.kpi-label {
    font-size: 11px;
    color: rgba(255,255,255,0.65);
    font-weight: 500;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Section headers ── */
.section-header {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #60a5fa;
    margin: 0 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(96,165,250,0.25);
}

/* ── Model cards ── */
.model-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
}
.model-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 18px;
}
.model-card.primary {
    border-color: #2563eb;
    background: #0f2851;
}
.model-card-name {
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.model-auc {
    font-size: 22px;
    font-weight: 700;
    color: #60a5fa;
}
.model-sub {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 4px;
}
.badge-primary {
    background: #1d4ed8;
    color: #bfdbfe;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 8px;
    font-weight: 600;
}

/* ── Risk output ── */
.risk-box {
    border-radius: 14px;
    padding: 24px 28px;
    text-align: center;
    margin: 16px 0;
}
.risk-box.low    { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #10b981; }
.risk-box.medium { background: linear-gradient(135deg, #78350f, #92400e); border: 1px solid #f59e0b; }
.risk-box.high   { background: linear-gradient(135deg, #7f1d1d, #991b1b); border: 1px solid #ef4444; }
.risk-box .risk-pct {
    font-size: 48px;
    font-weight: 800;
    color: white;
    line-height: 1;
}
.risk-box .risk-label {
    font-size: 16px;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    margin-top: 6px;
}
.risk-box .risk-advice {
    font-size: 13px;
    color: rgba(255,255,255,0.7);
    margin-top: 10px;
    line-height: 1.5;
}

/* ── Info panel ── */
.info-panel {
    background: #0f172a;
    border: 1px solid #1e3a8a;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.info-panel p {
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.6;
    margin: 0;
}
.info-panel strong {
    color: #e2e8f0;
}

/* ── Prob bar ── */
.prob-bar-wrap { margin: 12px 0; }
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 4px;
}
.prob-bar-label span:last-child { color: #60a5fa; font-weight: 600; }
.prob-bar-bg {
    background: #1e293b;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}

/* ── Data table ── */
.summary-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.summary-table th {
    background: #1e3a8a;
    color: #bfdbfe;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.summary-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #1e293b;
    color: #e2e8f0;
}
.summary-table tr:hover td { background: #1e293b; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0f1e !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 12px;
    font-weight: 500;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    width: 100% !important;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.5);
}

/* ── Divider ── */
hr { border-color: #1e293b !important; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ── Load & preprocess REAL dataset ────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_uci.csv")
    df["target"] = (df["num"] > 0).astype(int)
    df["sex_enc"]     = (df["sex"] == "Male").astype(int)
    df["cp_enc"]      = df["cp"].map({"typical angina": 0, "atypical angina": 1, "non-anginal": 2, "asymptomatic": 3})
    df["fbs_enc"]     = df["fbs"].map({True: 1, False: 0, "True": 1, "False": 0})
    df["restecg_enc"] = df["restecg"].map({"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2})
    df["exang_enc"]   = df["exang"].map({True: 1, False: 0, "True": 1, "False": 0})
    df["slope_enc"]   = df["slope"].map({"upsloping": 0, "flat": 1, "downsloping": 2})
    df["thal_enc"]    = df["thal"].map({"normal": 1, "fixed defect": 2, "reversable defect": 3})
    features = ["age","sex_enc","cp_enc","trestbps","chol","fbs_enc","restecg_enc",
                "thalch","exang_enc","oldpeak","slope_enc","ca","thal_enc"]
    X = df[features]
    y = df["target"]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=features)
    return df, X_imp, y, features, imputer

@st.cache_resource
def train_models(_X, _y):
    X_tr, X_te, y_tr, y_te = train_test_split(_X, _y, test_size=0.2, random_state=42, stratify=_y)
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    lr.fit(X_tr_s, y_tr)
    lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te_s)[:,1])
    spos = (y_tr==0).sum()/(y_tr==1).sum()
    xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spos,
        use_label_encoder=False, eval_metric="logloss", random_state=42)
    xgb_m.fit(X_tr, y_tr)
    xgb_auc = roc_auc_score(y_te, xgb_m.predict_proba(X_te)[:,1])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(xgb_m, _X, _y, cv=cv, scoring="roc_auc").mean()
    return lr, xgb_m, scaler, lr_auc, xgb_auc, cv_auc, X_te, y_te, X_te_s

df_raw, X, y, features, imputer = load_data()
lr_model, xgb_model, scaler, lr_auc, xgb_auc, xgb_cv, X_test, y_test, X_test_s = train_models(X, y)

FEAT_LABELS = {
    "age":"Age","sex_enc":"Sex","cp_enc":"Chest Pain","trestbps":"Resting BP",
    "chol":"Cholesterol","fbs_enc":"Fasting BS","restecg_enc":"Resting ECG",
    "thalch":"Max Heart Rate","exang_enc":"Exercise Angina","oldpeak":"ST Depression",
    "slope_enc":"ST Slope","ca":"Major Vessels","thal_enc":"Thalassemia"
}

# ── matplotlib dark theme ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f172a",
    "axes.facecolor":    "#0f172a",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e293b",
    "grid.linewidth":    0.6,
})

BLUE  = "#3b82f6"
RED   = "#ef4444"
GREEN = "#10b981"
AMBER = "#f59e0b"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:36px'>🫀</div>
        <div style='font-size:16px; font-weight:700; color:#e2e8f0;'>CardioRisk AI</div>
        <div style='font-size:11px; color:#64748b; margin-top:4px;'>Patient Input Panel</div>
    </div>
    """, unsafe_allow_html=True)

    age      = st.slider("Age (years)", int(df_raw.age.min()), int(df_raw.age.max()), 54)
    sex      = st.selectbox("Sex", ["Male", "Female"])
    cp       = st.selectbox("Chest Pain Type", ["Asymptomatic","Typical Angina","Atypical Angina","Non-Anginal"])
    trestbps = st.slider("Resting BP (mmHg)", 90, 200, 130)
    chol     = st.slider("Cholesterol (mg/dL)", 100, 600, 240)
    fbs      = st.radio("Fasting Blood Sugar > 120 mg/dL", ["No","Yes"], horizontal=True)
    restecg  = st.selectbox("Resting ECG", ["Normal","ST-T Abnormality","LV Hypertrophy"])
    thalch   = st.slider("Max Heart Rate", 70, 210, 150)
    exang    = st.radio("Exercise-Induced Angina", ["No","Yes"], horizontal=True)
    oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope    = st.selectbox("ST Slope", ["Upsloping","Flat","Downsloping"])
    ca       = st.slider("Major Vessels (0–3)", 0, 3, 0)
    thal     = st.selectbox("Thalassemia", ["Normal","Fixed Defect","Reversable Defect"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Predict Cardiac Risk")
    st.markdown(f"<div style='font-size:11px;color:#475569;text-align:center;margin-top:12px;'>Model trained on {len(df_raw)} UCI patients</div>", unsafe_allow_html=True)

def encode_inputs():
    cp_map    = {"Typical Angina":0,"Atypical Angina":1,"Non-Anginal":2,"Asymptomatic":3}
    ecg_map   = {"Normal":0,"ST-T Abnormality":1,"LV Hypertrophy":2}
    slope_map = {"Upsloping":0,"Flat":1,"Downsloping":2}
    thal_map  = {"Normal":1,"Fixed Defect":2,"Reversable Defect":3}
    return pd.DataFrame([[age, 1 if sex=="Male" else 0, cp_map[cp], trestbps, chol,
        1 if fbs=="Yes" else 0, ecg_map[restecg], thalch, 1 if exang=="Yes" else 0,
        oldpeak, slope_map[slope], ca, thal_map[thal]]], columns=features)

# ── MAIN ───────────────────────────────────────────────────────────────────────
# Hero
st.markdown(f"""
<div class="hero-banner">
    <h1>🫀 CardioRisk AI</h1>
    <p>Heart Disease Risk Predictor — XGBoost + Logistic Regression Ensemble</p>
    <div class="hero-badges">
        <span class="hero-badge">UCI Heart Disease Dataset</span>
        <span class="hero-badge">920 Real Patients</span>
        <span class="hero-badge">AUC-ROC {xgb_auc:.3f}</span>
        <span class="hero-badge">4 Clinical Sites</span>
        <span class="hero-badge">Educational Use Only</span>
    </div>
</div>
""", unsafe_allow_html=True)

# KPI row
disease_pct = y.mean() * 100
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-value">{len(df_raw)}</div>
        <div class="kpi-label">Total Patients</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{xgb_auc:.3f}</div>
        <div class="kpi-label">XGBoost AUC-ROC</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{xgb_cv:.3f}</div>
        <div class="kpi-label">5-Fold CV AUC</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{disease_pct:.0f}%</div>
        <div class="kpi-label">Disease Prevalence</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Dashboard row 1: Models | Charts | Info ────────────────────────────────────
col1, col2, col3 = st.columns([1, 1.6, 1])

with col1:
    st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-card primary" style="margin-bottom:10px;">
        <div class="model-card-name">XGBoost <span class="badge-primary">PRIMARY</span></div>
        <div class="model-auc">{xgb_auc:.3f}</div>
        <div class="model-sub">AUC-ROC &nbsp;·&nbsp; 5-Fold CV: {xgb_cv:.3f}</div>
        <div class="model-sub" style="margin-top:6px;">Weight in ensemble: 65%</div>
    </div>
    <div class="model-card">
        <div class="model-card-name">Logistic Regression</div>
        <div class="model-auc">{lr_auc:.3f}</div>
        <div class="model-sub">AUC-ROC &nbsp;·&nbsp; Interpretable baseline</div>
        <div class="model-sub" style="margin-top:6px;">Weight in ensemble: 35%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))

    # Class balance
    vals = y.value_counts().sort_index()
    axes[0].pie([vals[0], vals[1]], labels=["No Disease","Disease"],
                colors=[BLUE, RED], autopct="%1.0f%%", startangle=90,
                textprops={"fontsize":8, "color":"#e2e8f0"},
                wedgeprops={"linewidth":2,"edgecolor":"#0f172a"})
    axes[0].set_title("Class Balance", fontsize=9, fontweight="600", color="#e2e8f0")

    # Age by outcome
    axes[1].hist(df_raw[df_raw["target"]==0]["age"], bins=18, alpha=0.75, color=BLUE, label="No Disease")
    axes[1].hist(df_raw[df_raw["target"]==1]["age"], bins=18, alpha=0.75, color=RED, label="Disease")
    axes[1].set_title("Age by Outcome", fontsize=9, fontweight="600", color="#e2e8f0")
    axes[1].set_xlabel("Age", fontsize=8); axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.4)

    # Source sites
    src = df_raw["dataset"].value_counts()
    bars = axes[2].bar(src.index, src.values,
                       color=[BLUE, GREEN, AMBER, "#8b5cf6"][:len(src)],
                       edgecolor="#0f172a", linewidth=1.5, width=0.55)
    axes[2].set_title("Source Sites", fontsize=9, fontweight="600", color="#e2e8f0")
    axes[2].tick_params(axis="x", rotation=20, labelsize=7)
    axes[2].grid(axis="y", alpha=0.4)
    for bar in bars:
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                     str(int(bar.get_height())), ha="center", fontsize=8, color="#94a3b8")

    for ax in axes: ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col3:
    st.markdown('<p class="section-header">Quick Reference</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-panel">
        <p>
        <strong>How to use:</strong><br>
        Fill all patient vitals in the sidebar, then click <strong>Predict Cardiac Risk</strong>.<br><br>
        <strong>Features:</strong> 13 clinical variables from the UCI Heart Disease dataset, covering demographics, vitals, ECG findings, and stress test results.<br><br>
        <strong>Missing data:</strong> Median imputation applied (ca: 66%, thal: 53%, slope: 34% missing).
        </p>
    </div>
    <div class="info-panel" style="border-color:#7f1d1d; background:#1c0a0a;">
        <p style="color:#fca5a5;">
        ⚠️ <strong style="color:#f87171;">Not a clinical tool.</strong><br>
        For research & educational purposes only. Do not use for clinical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── PREDICTION OUTPUT ──────────────────────────────────────────────────────────
if predict_btn:
    X_in     = encode_inputs()
    X_in_imp = pd.DataFrame(imputer.transform(X_in), columns=features)
    xgb_prob = xgb_model.predict_proba(X_in_imp)[0][1]
    lr_prob  = lr_model.predict_proba(scaler.transform(X_in_imp))[0][1]
    ens_prob = xgb_prob * 0.65 + lr_prob * 0.35

    if ens_prob < 0.30:
        risk_cls, risk_lbl, advice = "low", "🟢 LOW RISK", "Routine monitoring recommended. Maintain a healthy lifestyle with regular exercise and balanced diet."
    elif ens_prob < 0.60:
        risk_cls, risk_lbl, advice = "medium", "🟡 MODERATE RISK", "Further evaluation strongly advised. Consider stress test, lipid panel, and echocardiogram."
    else:
        risk_cls, risk_lbl, advice = "high", "🔴 HIGH RISK", "Immediate cardiology referral recommended. Urgent evaluation and intervention may be required."

    r1, r2 = st.columns([1, 1.8])

    with r1:
        st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-box {risk_cls}">
            <div class="risk-pct">{ens_prob*100:.1f}%</div>
            <div class="risk-label">{risk_lbl}</div>
            <div class="risk-advice">{advice}</div>
        </div>
        """, unsafe_allow_html=True)

        # Model probability bars
        st.markdown("""
        <div class="info-panel" style="margin-top:14px;">
        """, unsafe_allow_html=True)
        for name, prob, color in [("XGBoost", xgb_prob, "#3b82f6"), ("Logistic Regression", lr_prob, "#8b5cf6"), ("Ensemble", ens_prob, "#10b981")]:
            st.markdown(f"""
            <div class="prob-bar-wrap">
                <div class="prob-bar-label"><span>{name}</span><span>{prob*100:.1f}%</span></div>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{prob*100:.1f}%;background:{color};"></div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown('<p class="section-header">Feature Importance — XGBoost</p>', unsafe_allow_html=True)
        imp = xgb_model.feature_importances_
        fi  = pd.DataFrame({"Feature":[FEAT_LABELS[f] for f in features],"Importance":imp}).sort_values("Importance",ascending=True)

        fig2, ax = plt.subplots(figsize=(8, 4.8))
        colors   = [RED if v > 0.09 else BLUE for v in fi["Importance"]]
        bars     = ax.barh(fi["Feature"], fi["Importance"], color=colors, edgecolor="#0f172a", linewidth=1, height=0.62)
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.set_title("Which features drive the prediction?", fontsize=11, fontweight="600", color="#e2e8f0", pad=10)
        ax.axvline(x=0.09, color="#475569", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, fi["Importance"]):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8.5, color="#94a3b8")
        p1 = mpatches.Patch(color=RED,  label="High importance (>9%)")
        p2 = mpatches.Patch(color=BLUE, label="Standard importance")
        ax.legend(handles=[p1,p2], fontsize=8, loc="lower right", framealpha=0.2)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Bottom row: confusion matrix + patient table
    b1, b2 = st.columns([1.2, 1])

    with b1:
        st.markdown('<p class="section-header">Confusion Matrix — Test Set</p>', unsafe_allow_html=True)
        fig3, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for i, (mname, preds) in enumerate([
            ("XGBoost", xgb_model.predict(X_test)),
            ("Logistic Regression", lr_model.predict(X_test_s))
        ]):
            cm = confusion_matrix(y_test, preds)
            im = axes[i].imshow(cm, cmap="Blues", vmin=0)
            axes[i].set_title(mname, fontsize=10, fontweight="600", color="#e2e8f0")
            axes[i].set_xlabel("Predicted", fontsize=8, color="#94a3b8")
            axes[i].set_ylabel("Actual", fontsize=8, color="#94a3b8")
            axes[i].set_xticks([0,1]); axes[i].set_yticks([0,1])
            axes[i].set_xticklabels(["No Disease","Disease"], fontsize=7.5)
            axes[i].set_yticklabels(["No Disease","Disease"], fontsize=7.5)
            for r in range(2):
                for c in range(2):
                    axes[i].text(c, r, str(cm[r,c]), ha="center", va="center",
                                 fontsize=16, fontweight="700",
                                 color="white" if cm[r,c] > cm.max()*0.5 else "#1e293b")
        plt.tight_layout(pad=1.5)
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    with b2:
        st.markdown('<p class="section-header">Patient Profile Summary</p>', unsafe_allow_html=True)
        rows = [
            ("Age", age), ("Sex", sex), ("Chest Pain", cp),
            ("Resting BP", f"{trestbps} mmHg"), ("Cholesterol", f"{chol} mg/dL"),
            ("Fasting BS >120", fbs), ("ECG", restecg),
            ("Max Heart Rate", thalch), ("Exercise Angina", exang),
            ("ST Depression", oldpeak), ("ST Slope", slope),
            ("Major Vessels", ca), ("Thalassemia", thal),
        ]
        tbl = "".join(f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k,v in rows)
        st.markdown(f"""
        <table class="summary-table">
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>{tbl}</tbody>
        </table>
        """, unsafe_allow_html=True)

else:
    # EDA section
    st.markdown('<p class="section-header">Exploratory Analysis</p>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        cp_counts = df_raw.groupby(["cp","target"]).size().unstack(fill_value=0)
        cp_counts.plot(kind="bar", ax=ax, color=[BLUE, RED], edgecolor="#0f172a", linewidth=1.2, width=0.65)
        ax.set_title("Chest Pain Type vs Disease Presence", fontsize=11, fontweight="600", color="#e2e8f0")
        ax.set_xlabel(""); ax.set_ylabel("Patient Count", fontsize=9)
        ax.legend(["No Disease","Disease"], fontsize=8, framealpha=0.2)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    with e2:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.hist(df_raw[df_raw["target"]==0]["thalch"].dropna(), bins=22, alpha=0.75, color=BLUE, label="No Disease")
        ax.hist(df_raw[df_raw["target"]==1]["thalch"].dropna(), bins=22, alpha=0.75, color=RED, label="Disease")
        ax.set_title("Max Heart Rate by Outcome", fontsize=11, fontweight="600", color="#e2e8f0")
        ax.set_xlabel("Max Heart Rate (bpm)", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.2)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    st.markdown("""
    <div style='text-align:center; padding:20px; color:#475569; font-size:14px;'>
        👈 &nbsp; Fill patient vitals in the sidebar and click <strong style='color:#60a5fa;'>Predict Cardiac Risk</strong>
    </div>
    """, unsafe_allow_html=True)

# Footer
import streamlit.components.v1 as components
components.html("""
<div style='margin-top:32px; border-top:1px solid #1e293b; padding-top:28px; font-family: Inter, sans-serif;'>

    <!-- App credit line -->
    <div style='text-align:center; margin-bottom:24px;'>
        <span style='font-size:12px; color:#334155;'>
            CardioRisk AI &nbsp;·&nbsp; XGBoost + Logistic Regression Ensemble &nbsp;·&nbsp;
            UCI Heart Disease Dataset (920 patients) &nbsp;·&nbsp;
            <strong style='color:#475569;'>For research & educational purposes only — not a medical device</strong>
        </span>
    </div>

    <!-- Developer card -->
    <div style='
        max-width: 520px;
        margin: 0 auto 20px auto;
        background: linear-gradient(135deg, #0f172a, #0f2851);
        border: 1px solid #1e3a8a;
        border-radius: 16px;
        padding: 24px 28px;
        text-align: center;
    '>
        <div style='font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:#3b82f6; margin-bottom:10px;'>Built by</div>
        <div style='font-size:22px; font-weight:800; color:#e2e8f0; margin-bottom:4px; letter-spacing:-0.3px;'>Pranshu Kumar</div>
        <div style='font-size:12px; color:#64748b; margin-bottom:20px;'>ML Engineer &nbsp;·&nbsp; Healthcare AI</div>

        <div style='display:flex; justify-content:center; gap:12px; flex-wrap:wrap;'>

            <a href='https://github.com/dev-pranshu04' target='_blank' style='
                display:inline-flex; align-items:center; gap:7px;
                background:#161b22; border:1px solid #30363d;
                color:#e2e8f0; font-size:12px; font-weight:500;
                padding:8px 16px; border-radius:8px; text-decoration:none;
            '>
                <svg width='15' height='15' viewBox='0 0 24 24' fill='#e2e8f0'>
                    <path d='M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.38.6.1.82-.26.82-.58v-2.03c-3.34.72-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.74.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.83 2.8 1.3 3.49 1 .1-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.14-.3-.54-1.52.1-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02 0 2.04.13 3 .4 2.28-1.55 3.29-1.23 3.29-1.23.65 1.66.24 2.88.12 3.18.77.84 1.23 1.91 1.23 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.69.83.57C20.57 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z'/>
                </svg>
                GitHub
            </a>

            <a href='https://www.linkedin.com/in/dev-pranshu' target='_blank' style='
                display:inline-flex; align-items:center; gap:7px;
                background:#0a66c2; border:1px solid #0a66c2;
                color:white; font-size:12px; font-weight:500;
                padding:8px 16px; border-radius:8px; text-decoration:none;
            '>
                <svg width='15' height='15' viewBox='0 0 24 24' fill='white'>
                    <path d='M20.45 20.45h-3.55v-5.57c0-1.33-.03-3.04-1.85-3.04-1.85 0-2.13 1.45-2.13 2.94v5.67H9.37V9h3.41v1.56h.05c.47-.9 1.63-1.85 3.36-1.85 3.59 0 4.26 2.36 4.26 5.44v6.3zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.55V9h3.57v11.45zM22.23 0H1.77C.79 0 0 .77 0 1.72v20.56C0 23.23.79 24 1.77 24h20.46c.98 0 1.77-.77 1.77-1.72V1.72C24 .77 23.21 0 22.23 0z'/>
                </svg>
                LinkedIn
            </a>

            <a href='https://www.instagram.com/im_pranshu29/' target='_blank' style='
                display:inline-flex; align-items:center; gap:7px;
                background: linear-gradient(135deg, #833ab4, #fd1d1d, #fcb045);
                border:none;
                color:white; font-size:12px; font-weight:500;
                padding:8px 16px; border-radius:8px; text-decoration:none;
            '>
                <svg width='15' height='15' viewBox='0 0 24 24' fill='white'>
                    <path d='M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838a6.162 6.162 0 1 0 0 12.324 6.162 6.162 0 0 0 0-12.324zM12 16a4 4 0 1 1 0-8 4 4 0 0 1 0 8zm6.406-11.845a1.44 1.44 0 1 0 0 2.881 1.44 1.44 0 0 0 0-2.881z'/>
                </svg>
                Instagram
            </a>

        </div>
    </div>

</div>
""", height=180)
