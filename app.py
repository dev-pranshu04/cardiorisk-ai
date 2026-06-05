import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .risk-low    { background:#d1fae5; color:#065f46; border-radius:12px; padding:20px; text-align:center; font-size:26px; font-weight:700; }
    .risk-medium { background:#fef3c7; color:#92400e; border-radius:12px; padding:20px; text-align:center; font-size:26px; font-weight:700; }
    .risk-high   { background:#fee2e2; color:#991b1b; border-radius:12px; padding:20px; text-align:center; font-size:26px; font-weight:700; }
    .metric-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 18px; margin:6px 0; }
    .stButton > button { background:#1e3a5f; color:white; border:none; border-radius:8px; padding:10px 28px; font-weight:600; width:100%; font-size:16px; }
    .stButton > button:hover { background:#2563eb; }
    h1 { color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ── Load & preprocess REAL dataset ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_uci.csv")

    # Binary target: 0 = no disease, 1 = disease present (num > 0)
    df["target"] = (df["num"] > 0).astype(int)

    # Encode categorical columns
    df["sex_enc"]     = (df["sex"] == "Male").astype(int)
    df["cp_enc"]      = df["cp"].map({
        "typical angina": 0, "atypical angina": 1,
        "non-anginal": 2, "asymptomatic": 3
    })
    df["fbs_enc"]     = df["fbs"].map({True: 1, False: 0, "True": 1, "False": 0})
    df["restecg_enc"] = df["restecg"].map({
        "normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2
    })
    df["exang_enc"]   = df["exang"].map({True: 1, False: 0, "True": 1, "False": 0})
    df["slope_enc"]   = df["slope"].map({"upsloping": 0, "flat": 1, "downsloping": 2})
    df["thal_enc"]    = df["thal"].map({"normal": 1, "fixed defect": 2, "reversable defect": 3})

    features = ["age", "sex_enc", "cp_enc", "trestbps", "chol",
                "fbs_enc", "restecg_enc", "thalch", "exang_enc",
                "oldpeak", "slope_enc", "ca", "thal_enc"]

    X = df[features]
    y = df["target"]

    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

    return df, X_imputed, y, features, imputer

@st.cache_resource
def train_models(_X, _y):
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])

    # XGBoost
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_m = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42
    )
    xgb_m.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb_m.predict_proba(X_test)[:, 1])

    # Cross-val AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_cv = cross_val_score(xgb_m, _X, _y, cv=cv, scoring="roc_auc").mean()

    return lr, xgb_m, scaler, lr_auc, xgb_auc, xgb_cv, X_test, y_test, scaler.transform(X_test)

df_raw, X, y, features, imputer = load_data()
lr_model, xgb_model, scaler, lr_auc, xgb_auc, xgb_cv, X_test, y_test, X_test_s = train_models(X, y)

FEATURE_LABELS = {
    "age": "Age", "sex_enc": "Sex", "cp_enc": "Chest Pain Type",
    "trestbps": "Resting BP", "chol": "Cholesterol",
    "fbs_enc": "Fasting Blood Sugar", "restecg_enc": "Resting ECG",
    "thalch": "Max Heart Rate", "exang_enc": "Exercise Angina",
    "oldpeak": "ST Depression", "slope_enc": "ST Slope",
    "ca": "Major Vessels", "thal_enc": "Thalassemia"
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🫀 Patient Vitals")
st.sidebar.caption(f"Model trained on {len(df_raw)} real UCI patients")
st.sidebar.markdown("---")

age      = st.sidebar.slider("Age", int(df_raw.age.min()), int(df_raw.age.max()), 54)
sex      = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp       = st.sidebar.selectbox("Chest Pain Type", [
    "Asymptomatic", "Typical Angina", "Atypical Angina", "Non-Anginal"])
trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
chol     = st.sidebar.slider("Cholesterol (mg/dL)", 100, 600, 240)
fbs      = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
restecg  = st.sidebar.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
thalch   = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
exang    = st.sidebar.radio("Exercise-Induced Angina", ["No", "Yes"])
oldpeak  = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, 0.1)
slope    = st.sidebar.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
ca       = st.sidebar.slider("Major Vessels Coloured (0–3)", 0, 3, 0)
thal     = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])

predict_btn = st.sidebar.button("🔍 Predict Risk")

def encode_inputs():
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal": 2, "Asymptomatic": 3}
    ecg_map = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}
    return pd.DataFrame([[
        age,
        1 if sex == "Male" else 0,
        cp_map[cp],
        trestbps, chol,
        1 if fbs == "Yes" else 0,
        ecg_map[restecg],
        thalch,
        1 if exang == "Yes" else 0,
        oldpeak,
        slope_map[slope],
        ca,
        thal_map[thal]
    ]], columns=features)

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🫀 CardioRisk AI — Heart Disease Risk Predictor")
st.markdown(f"*Trained on **{len(df_raw)} real UCI Heart Disease patients** (Cleveland · VA · Hungarian · Switzerland)*")
st.markdown("---")

col1, col2, col3 = st.columns([1.1, 1.3, 0.9])

with col1:
    st.subheader("📊 Model Performance")
    st.markdown(f"""
    <div class="metric-card">
        <b>XGBoost</b> ⭐ Primary<br>
        AUC-ROC: <b>{xgb_auc:.3f}</b> &nbsp;|&nbsp; 5-Fold CV: <b>{xgb_cv:.3f}</b>
    </div>
    <div class="metric-card">
        <b>Logistic Regression</b><br>
        AUC-ROC: <b>{lr_auc:.3f}</b>
    </div>
    <div class="metric-card">
        Training: <b>{int(len(X)*0.8)}</b> patients &nbsp;|&nbsp; Test: <b>{int(len(X)*0.2)}</b><br>
        Disease prevalence: <b>{y.mean()*100:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("🧬 Dataset Overview")
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    # Class balance
    vals = df_raw["target"].value_counts()
    axes[0].pie(vals, labels=["Disease", "No Disease"] if vals.index[0]==1 else ["No Disease","Disease"],
                colors=["#ef4444","#3b82f6"] if vals.index[0]==1 else ["#3b82f6","#ef4444"],
                autopct="%1.0f%%", startangle=90, textprops={"fontsize":8})
    axes[0].set_title("Class Balance", fontsize=9, fontweight="bold")

    # Age distribution by class
    axes[1].hist(df_raw[df_raw["target"]==0]["age"], bins=15, alpha=0.6, color="#3b82f6", label="No Disease")
    axes[1].hist(df_raw[df_raw["target"]==1]["age"], bins=15, alpha=0.6, color="#ef4444", label="Disease")
    axes[1].set_title("Age by Outcome", fontsize=9, fontweight="bold")
    axes[1].legend(fontsize=7)
    axes[1].tick_params(labelsize=7)

    # Dataset source breakdown
    src = df_raw["dataset"].value_counts()
    axes[2].bar(src.index, src.values, color=["#3b82f6","#10b981","#f59e0b","#8b5cf6"], edgecolor="white")
    axes[2].set_title("Source Sites", fontsize=9, fontweight="bold")
    axes[2].tick_params(axis='x', labelsize=6, rotation=15)
    axes[2].tick_params(axis='y', labelsize=7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col3:
    st.subheader("ℹ️ Instructions")
    st.info("1. Fill patient vitals in the **sidebar**\n2. Click **Predict Risk**\n3. Review risk score + feature importance")
    st.warning("⚠️ Educational use only — not a clinical tool.")

    missing_pct = (df_raw[["ca","thal","slope"]].isnull().sum() / len(df_raw) * 100)
    st.caption(f"Missing data handled via median imputation.\nca: {missing_pct['ca']:.0f}% | thal: {missing_pct['thal']:.0f}% | slope: {missing_pct['slope']:.0f}%")

st.markdown("---")

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    X_in = encode_inputs()

    # Impute (in case any median fill needed — consistent with training)
    X_in_imp = pd.DataFrame(imputer.transform(X_in), columns=features)

    xgb_prob = xgb_model.predict_proba(X_in_imp)[0][1]
    lr_prob  = lr_model.predict_proba(scaler.transform(X_in_imp))[0][1]
    ens_prob = xgb_prob * 0.65 + lr_prob * 0.35

    c1, c2, c3 = st.columns(3)
    c1.metric("XGBoost", f"{xgb_prob*100:.1f}%")
    c2.metric("Logistic Regression", f"{lr_prob*100:.1f}%")
    c3.metric("Ensemble Score", f"{ens_prob*100:.1f}%")

    if ens_prob < 0.30:
        label, css, advice = "🟢 LOW RISK", "risk-low", "Routine monitoring recommended. Maintain healthy lifestyle."
    elif ens_prob < 0.60:
        label, css, advice = "🟡 MODERATE RISK", "risk-medium", "Further evaluation advised. Consider stress test & lipid panel."
    else:
        label, css, advice = "🔴 HIGH RISK", "risk-high", "Immediate cardiology referral strongly recommended."

    st.markdown(f'<div class="{css}">{label} — {ens_prob*100:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f"**Clinical note:** {advice}")
    st.markdown("---")

    # Feature importance
    st.subheader("🔬 Feature Importance (XGBoost — trained on your data)")
    imp = xgb_model.feature_importances_
    fi = pd.DataFrame({"Feature": [FEATURE_LABELS[f] for f in features], "Importance": imp}).sort_values("Importance", ascending=True)

    fig2, ax = plt.subplots(figsize=(9, 5))
    colors = ["#ef4444" if v > 0.09 else "#3b82f6" for v in fi["Importance"]]
    bars = ax.barh(fi["Feature"], fi["Importance"], color=colors, edgecolor="white", linewidth=0.5, height=0.65)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Which clinical features drive this model's predictions?", fontsize=12, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, fi["Importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    p1 = mpatches.Patch(color="#ef4444", label="High importance (>9%)")
    p2 = mpatches.Patch(color="#3b82f6", label="Standard importance")
    ax.legend(handles=[p1, p2], fontsize=9, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Confusion matrix
    st.subheader("📉 Model Confusion Matrix (Test Set)")
    fig3, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for idx, (model_name, preds) in enumerate([
        ("XGBoost", xgb_model.predict(X_test)),
        ("Logistic Regression", lr_model.predict(X_test_s))
    ]):
        cm = confusion_matrix(y_test, preds)
        im = axes[idx].imshow(cm, cmap="Blues")
        axes[idx].set_title(model_name, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted", fontsize=9)
        axes[idx].set_ylabel("Actual", fontsize=9)
        axes[idx].set_xticks([0,1]); axes[idx].set_yticks([0,1])
        axes[idx].set_xticklabels(["No Disease","Disease"], fontsize=8)
        axes[idx].set_yticklabels(["No Disease","Disease"], fontsize=8)
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, str(cm[i,j]), ha="center", va="center",
                               fontsize=14, fontweight="bold",
                               color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Patient summary table
    st.subheader("👤 Patient Input Summary")
    summary = {
        "Age": age, "Sex": sex, "Chest Pain": cp, "BP": f"{trestbps} mmHg",
        "Cholesterol": f"{chol} mg/dL", "Fasting BS >120": fbs,
        "ECG": restecg, "Max HR": thalch, "Exercise Angina": exang,
        "ST Depression": oldpeak, "ST Slope": slope,
        "Major Vessels": ca, "Thalassemia": thal
    }
    st.dataframe(pd.DataFrame(summary.items(), columns=["Parameter", "Value"]),
                 use_container_width=True, hide_index=True)

else:
    # EDA section when no prediction yet
    st.subheader("📈 Exploratory Data Analysis — Your UCI Dataset")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        cp_counts = df_raw.groupby(["cp","target"]).size().unstack(fill_value=0)
        cp_counts.plot(kind="bar", ax=ax, color=["#3b82f6","#ef4444"], edgecolor="white", width=0.7)
        ax.set_title("Chest Pain Type vs Disease Presence", fontsize=11, fontweight="bold")
        ax.set_xlabel("Chest Pain Type", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(["No Disease","Disease"], fontsize=8)
        ax.tick_params(axis='x', rotation=20, labelsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        d0 = df_raw[df_raw["target"]==0]["thalch"].dropna()
        d1 = df_raw[df_raw["target"]==1]["thalch"].dropna()
        ax.hist(d0, bins=20, alpha=0.6, color="#3b82f6", label="No Disease")
        ax.hist(d1, bins=20, alpha=0.6, color="#ef4444", label="Disease")
        ax.set_title("Max Heart Rate by Outcome", fontsize=11, fontweight="bold")
        ax.set_xlabel("Max Heart Rate (bpm)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("👈 Enter patient vitals in the sidebar and click **Predict Risk** to get a personalized risk score.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#6b7280;font-size:13px;'>"
    "CardioRisk AI • XGBoost + Logistic Regression • UCI Heart Disease Dataset (920 patients) • "
    "<b>For research & educational purposes only</b></p>",
    unsafe_allow_html=True
)
