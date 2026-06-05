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
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="CardioRisk AI", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.hero {
 background: linear-gradient(135deg, #0a0f1e 0%, #0f2044 50%, #162d6e 100%);
 border-radius: 20px; padding: 36px 40px; margin-bottom: 32px;
 border: 1px solid rgba(59,130,246,0.2);
 position: relative; overflow: hidden;
}
.hero::before {
 content: ''; position: absolute; top: -60px; right: -60px;
 width: 220px; height: 220px; border-radius: 50%;
 background: radial-gradient(circle, rgba(37,99,235,0.15) 0%, transparent 70%);
}
.hero-eyebrow { font-size: 11px; font-weight: 600; letter-spacing: 2px;
 text-transform: uppercase; color: #3b82f6; margin-bottom: 10px; }
.hero h1 { font-size: 32px; font-weight: 800; color: #f8fafc;
 margin: 0 0 8px; letter-spacing: -0.8px; line-height: 1.1; }
.hero-sub { font-size: 14px; color: rgba(255,255,255,0.45); font-weight: 400; margin: 0; }

.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 32px; }
.kpi { background: #0d1424; border: 1px solid #1e2d4a; border-radius: 14px;
 padding: 20px 22px; position: relative; overflow: hidden; }
.kpi::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0;
 height: 2px; background: linear-gradient(90deg, #1d4ed8, #3b82f6); }
.kpi-val { font-size: 28px; font-weight: 700; color: #60a5fa; line-height: 1; }
.kpi-lbl { font-size: 11px; color: #475569; font-weight: 500;
 text-transform: uppercase; letter-spacing: 0.6px; margin-top: 6px; }

.sec-label { font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
 text-transform: uppercase; color: #2563eb; margin-bottom: 16px;
 padding-bottom: 8px; border-bottom: 1px solid rgba(37,99,235,0.2); }

.mcard { background: #0d1424; border: 1px solid #1e2d4a;
 border-radius: 14px; padding: 18px 20px; margin-bottom: 12px; }
.mcard.top { border-color: #1d4ed8; background: linear-gradient(135deg,#0d1f42,#0f2851); }
.mcard-name { font-size: 12px; font-weight: 600; color: #94a3b8;
 text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; display:flex; align-items:center; gap:8px; }
.chip { background: #1d4ed8; color: #bfdbfe; font-size: 9px; font-weight: 700;
 padding: 2px 8px; border-radius: 6px; letter-spacing: 0.5px; }
.mcard-auc { font-size: 30px; font-weight: 700; color: #f8fafc; line-height: 1; }
.mcard-meta { font-size: 11px; color: #334155; margin-top: 8px; }

.qref { background: #0d1424; border: 1px solid #1e2d4a;
 border-radius: 14px; padding: 20px; }
.qref p { font-size: 13px; color: #64748b; line-height: 1.7; margin: 0; }
.qref strong { color: #94a3b8; font-weight: 500; }

.risk-wrap { border-radius: 16px; padding: 28px 32px; text-align: center; margin-bottom: 20px; }
.risk-wrap.low  { background: linear-gradient(135deg,#052e16,#064e3b); border:1px solid #059669; }
.risk-wrap.med  { background: linear-gradient(135deg,#431407,#7c2d12); border:1px solid #d97706; }
.risk-wrap.high { background: linear-gradient(135deg,#450a0a,#7f1d1d); border:1px solid #dc2626; }
.risk-pct { font-size: 52px; font-weight: 800; color: #fff; line-height: 1; }
.risk-tag { font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.7);
 margin-top: 8px; letter-spacing: 0.5px; text-transform: uppercase; }
.risk-note { font-size: 12px; color: rgba(255,255,255,0.45);
 margin-top: 10px; line-height: 1.6; }

.pbar-row { margin: 10px 0; }
.pbar-top { display:flex; justify-content:space-between;
 font-size: 11px; color: #475569; margin-bottom: 5px; }
.pbar-top span:last-child { color: #60a5fa; font-weight: 600; }
.pbar-bg { background: #0f172a; border-radius: 4px; height: 5px; }
.pbar-fill { height: 100%; border-radius: 4px; }

.ptable { width:100%; border-collapse:collapse; font-size:12px; }
.ptable th { color: #334155; font-weight: 600; font-size: 10px; text-transform: uppercase;
 letter-spacing: 0.7px; padding: 6px 10px; background: #0a0f1e; }
.ptable td { padding: 7px 10px; border-bottom: 1px solid #0f172a; color: #94a3b8; }
.ptable td:first-child { color: #475569; }
.ptable tr:hover td { background: #0d1424; }

[data-testid="stSidebar"] { background: #070c18 !important; border-right: 1px solid #111827; }
[data-testid="stSidebar"] label { color: #475569 !important; font-size: 11px !important;
 font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.5px; }
.stButton > button {
 background: linear-gradient(135deg,#1d4ed8,#2563eb) !important;
 color: #fff !important; border: none !important; border-radius: 10px !important;
 padding: 13px !important; font-weight: 600 !important; font-size: 14px !important;
 width: 100% !important; letter-spacing: 0.3px !important;
 box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
}
hr { border-color: #111827 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
 df = pd.read_csv("heart_disease_uci.csv")
 df["target"]      = (df["num"] > 0).astype(int)
 df["sex_enc"]     = (df["sex"] == "Male").astype(int)
 df["cp_enc"]      = df["cp"].map({"typical angina":0,"atypical angina":1,"non-anginal":2,"asymptomatic":3})
 df["fbs_enc"]     = df["fbs"].map({True:1,False:0,"True":1,"False":0})
 df["restecg_enc"] = df["restecg"].map({"normal":0,"st-t abnormality":1,"lv hypertrophy":2})
 df["exang_enc"]   = df["exang"].map({True:1,False:0,"True":1,"False":0})
 df["slope_enc"]   = df["slope"].map({"upsloping":0,"flat":1,"downsloping":2})
 df["thal_enc"]    = df["thal"].map({"normal":1,"fixed defect":2,"reversable defect":3})
 feats = ["age","sex_enc","cp_enc","trestbps","chol","fbs_enc","restecg_enc",
          "thalch","exang_enc","oldpeak","slope_enc","ca","thal_enc"]
 X = df[feats]; y = df["target"]
 imp = SimpleImputer(strategy="median")
 X_imp = pd.DataFrame(imp.fit_transform(X), columns=feats)
 return df, X_imp, y, feats, imp

@st.cache_resource
def train(_X, _y):
 Xtr,Xte,ytr,yte = train_test_split(_X,_y,test_size=0.2,random_state=42,stratify=_y)
 sc = StandardScaler()
 Xtr_s,Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
 lr = LogisticRegression(max_iter=1000,C=0.5,random_state=42)
 lr.fit(Xtr_s,ytr)
 lr_auc = roc_auc_score(yte, lr.predict_proba(Xte_s)[:,1])
 sp = (ytr==0).sum()/(ytr==1).sum()
 xm = xgb.XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
     subsample=0.8,colsample_bytree=0.8,scale_pos_weight=sp,
     use_label_encoder=False,eval_metric="logloss",random_state=42)
 xm.fit(Xtr,ytr)
 xgb_auc = roc_auc_score(yte, xm.predict_proba(Xte)[:,1])
 cv_auc  = cross_val_score(xm,_X,_y,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring="roc_auc").mean()
 return lr, xm, sc, lr_auc, xgb_auc, cv_auc, Xte, yte, Xte_s

df_raw, X, y, feats, imputer = load_data()
lr_m, xgb_m, scaler, lr_auc, xgb_auc, cv_auc, Xte, yte, Xte_s = train(X, y)

FLABELS = {"age":"Age","sex_enc":"Sex","cp_enc":"Chest Pain","trestbps":"Resting BP",
        "chol":"Cholesterol","fbs_enc":"Fasting BS","restecg_enc":"ECG",
        "thalch":"Max Heart Rate","exang_enc":"Exercise Angina","oldpeak":"ST Depression",
        "slope_enc":"ST Slope","ca":"Major Vessels","thal_enc":"Thalassemia"}

plt.rcParams.update({"figure.facecolor":"#0d1424","axes.facecolor":"#0d1424",
 "axes.edgecolor":"#1e2d4a","axes.labelcolor":"#475569","xtick.color":"#334155",
 "ytick.color":"#334155","text.color":"#94a3b8","grid.color":"#111827","grid.linewidth":0.5})

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
 st.markdown("""
 <div style='text-align:center;padding:16px 0 24px;'>
     <div style='font-size:38px;margin-bottom:8px;'>🫀</div>
     <div style='font-size:15px;font-weight:700;color:#e2e8f0;'>CardioRisk AI</div>
     <div style='font-size:10px;color:#334155;margin-top:3px;letter-spacing:1px;text-transform:uppercase;'>Patient Input Panel</div>
 </div>""", unsafe_allow_html=True)

 age      = st.slider("Age (years)", int(df_raw.age.min()), int(df_raw.age.max()), 54)
 sex      = st.selectbox("Sex", ["Male","Female"])
 cp       = st.selectbox("Chest Pain", ["Asymptomatic","Typical Angina","Atypical Angina","Non-Anginal"])
 trestbps = st.slider("Resting BP (mmHg)", 90, 200, 130)
 chol     = st.slider("Cholesterol (mg/dL)", 100, 600, 240)
 fbs      = st.radio("Fasting BS > 120 mg/dL", ["No","Yes"], horizontal=True)
 restecg  = st.selectbox("Resting ECG", ["Normal","ST-T Abnormality","LV Hypertrophy"])
 thalch   = st.slider("Max Heart Rate", 70, 210, 150)
 exang    = st.radio("Exercise Angina", ["No","Yes"], horizontal=True)
 oldpeak  = st.slider("ST Depression", 0.0, 6.5, 1.0, 0.1)
 slope    = st.selectbox("ST Slope", ["Upsloping","Flat","Downsloping"])
 ca       = st.slider("Major Vessels (0–3)", 0, 3, 0)
 thal     = st.selectbox("Thalassemia", ["Normal","Fixed Defect","Reversable Defect"])
 st.markdown("<br>", unsafe_allow_html=True)
 go = st.button("🔍  Predict Cardiac Risk")

def encode():
 return pd.DataFrame([[age,1 if sex=="Male" else 0,
     {"Typical Angina":0,"Atypical Angina":1,"Non-Anginal":2,"Asymptomatic":3}[cp],
     trestbps,chol,1 if fbs=="Yes" else 0,
     {"Normal":0,"ST-T Abnormality":1,"LV Hypertrophy":2}[restecg],
     thalch,1 if exang=="Yes" else 0,oldpeak,
     {"Upsloping":0,"Flat":1,"Downsloping":2}[slope],ca,
     {"Normal":1,"Fixed Defect":2,"Reversable Defect":3}[thal]
 ]], columns=feats)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
 <div class="hero-eyebrow">Cardiovascular Risk Intelligence</div>
 <h1>🫀 CardioRisk AI</h1>
 <p class="hero-sub">XGBoost · Logistic Regression · UCI Heart Disease · {len(df_raw)} patients · 4 clinical sites</p>
</div>""", unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
 <div class="kpi"><div class="kpi-val">{xgb_auc:.3f}</div><div class="kpi-lbl">XGBoost AUC-ROC</div></div>
 <div class="kpi"><div class="kpi-val">{cv_auc:.3f}</div><div class="kpi-lbl">5-Fold CV AUC</div></div>
 <div class="kpi"><div class="kpi-val">{lr_auc:.3f}</div><div class="kpi-lbl">LR AUC-ROC</div></div>
 <div class="kpi"><div class="kpi-val">{y.mean()*100:.0f}%</div><div class="kpi-lbl">Disease Prevalence</div></div>
</div>""", unsafe_allow_html=True)

# ── Dashboard ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1, 1.7, 0.95])

with c1:
 st.markdown('<p class="sec-label">Models</p>', unsafe_allow_html=True)
 st.markdown(f"""
 <div class="mcard top">
     <div class="mcard-name">XGBoost <span class="chip">PRIMARY · 65%</span></div>
     <div class="mcard-auc">{xgb_auc:.3f}</div>
     <div class="mcard-meta">AUC-ROC &nbsp;·&nbsp; CV: {cv_auc:.3f}</div>
 </div>
 <div class="mcard">
     <div class="mcard-name">Logistic Regression <span style="color:#334155;font-size:9px;">35%</span></div>
     <div class="mcard-auc" style="font-size:24px;">{lr_auc:.3f}</div>
     <div class="mcard-meta">AUC-ROC &nbsp;·&nbsp; Interpretable baseline</div>
 </div>""", unsafe_allow_html=True)

with c2:
 st.markdown('<p class="sec-label">Dataset Overview</p>', unsafe_allow_html=True)
 fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
 B, R, G, P = "#3b82f6","#ef4444","#10b981","#8b5cf6"

 vals = y.value_counts().sort_index()
 axes[0].pie([vals[0],vals[1]], labels=["Healthy","Disease"], colors=[B,R],
     autopct="%1.0f%%", startangle=90, textprops={"fontsize":8,"color":"#94a3b8"},
     wedgeprops={"linewidth":2,"edgecolor":"#0d1424"})
 axes[0].set_title("Class Split", fontsize=9, fontweight="600", color="#64748b", pad=8)

 axes[1].hist(df_raw[df_raw["target"]==0]["age"],bins=18,alpha=0.8,color=B,label="Healthy",linewidth=0)
 axes[1].hist(df_raw[df_raw["target"]==1]["age"],bins=18,alpha=0.8,color=R,label="Disease",linewidth=0)
 axes[1].set_title("Age Distribution", fontsize=9, fontweight="600", color="#64748b", pad=8)
 axes[1].legend(fontsize=7, framealpha=0); axes[1].grid(axis="y",alpha=0.3)

 src = df_raw["dataset"].value_counts()
 clrs = [B,G,"#f59e0b",P][:len(src)]
 bars = axes[2].bar(src.index, src.values, color=clrs, edgecolor="#0d1424", linewidth=1.5, width=0.5)
 axes[2].set_title("Clinical Sites", fontsize=9, fontweight="600", color="#64748b", pad=8)
 axes[2].tick_params(axis="x", rotation=18, labelsize=6.5)
 axes[2].grid(axis="y", alpha=0.3)
 for b in bars:
     axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+2, str(int(b.get_height())),
         ha="center", fontsize=8, color="#475569")

 for ax in axes:
     ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
 plt.tight_layout(pad=1.0)
 st.pyplot(fig, use_container_width=True); plt.close()

with c3:
 st.markdown('<p class="sec-label">How to Use</p>', unsafe_allow_html=True)
 st.markdown("""
 <div class="qref">
     <p>
         Enter the patient's clinical vitals in the sidebar panel, then press
         <strong>Predict Cardiac Risk</strong>.<br><br>
         The ensemble combines XGBoost and Logistic Regression for a final probability score,
         along with a feature importance breakdown and model diagnostics.<br><br>
         <strong>⚠ Research use only.</strong><br>Not a certified medical device.
     </p>
 </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if go:
 Xi     = encode()
 Xi_imp = pd.DataFrame(imputer.transform(Xi), columns=feats)
 xp     = xgb_m.predict_proba(Xi_imp)[0][1]
 lp     = lr_m.predict_proba(scaler.transform(Xi_imp))[0][1]
 ep     = xp*0.65 + lp*0.35

 cls  = "low" if ep<0.30 else "med" if ep<0.60 else "high"
 tag  = ("LOW RISK" if ep<0.30 else "MODERATE RISK" if ep<0.60 else "HIGH RISK")
 note = ("Routine monitoring advised. Maintain healthy lifestyle habits."
         if ep<0.30 else
         "Further evaluation recommended — consider stress test and lipid panel."
         if ep<0.60 else
         "Cardiology referral indicated. Prompt clinical evaluation required.")

 left, right = st.columns([1, 1.8])

 with left:
     st.markdown('<p class="sec-label">Risk Score</p>', unsafe_allow_html=True)
     st.markdown(f"""
     <div class="risk-wrap {cls}">
         <div class="risk-pct">{ep*100:.1f}%</div>
         <div class="risk-tag">{tag}</div>
         <div class="risk-note">{note}</div>
     </div>""", unsafe_allow_html=True)

     st.markdown('<p class="sec-label" style="margin-top:20px;">Model Breakdown</p>', unsafe_allow_html=True)
     for nm, prob, col in [("XGBoost",xp,"#3b82f6"),("Logistic Regression",lp,"#8b5cf6"),("Ensemble",ep,"#10b981")]:
         st.markdown(f"""
         <div class="pbar-row">
             <div class="pbar-top"><span>{nm}</span><span>{prob*100:.1f}%</span></div>
             <div class="pbar-bg"><div class="pbar-fill" style="width:{prob*100:.1f}%;background:{col};"></div></div>
         </div>""", unsafe_allow_html=True)

 with right:
     st.markdown('<p class="sec-label">Feature Importance</p>', unsafe_allow_html=True)
     imp_vals = xgb_m.feature_importances_
     fi = pd.DataFrame({"f":[FLABELS[f] for f in feats],"v":imp_vals}).sort_values("v",ascending=True)

     fig2, ax = plt.subplots(figsize=(8,4.6))
     cols = ["#ef4444" if v>0.09 else "#3b82f6" for v in fi["v"]]
     bars = ax.barh(fi["f"], fi["v"], color=cols, edgecolor="#0d1424", linewidth=1, height=0.6)
     ax.set_xlabel("Importance", fontsize=9)
     ax.axvline(0.09, color="#1e2d4a", linestyle="--", linewidth=0.8)
     ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
     ax.grid(axis="x", alpha=0.25)
     for bar,val in zip(bars,fi["v"]):
         ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.3f}",
             va="center", fontsize=8, color="#334155")
     p1 = mpatches.Patch(color="#ef4444", label=">9% importance")
     p2 = mpatches.Patch(color="#3b82f6", label="Standard")
     ax.legend(handles=[p1,p2], fontsize=8, framealpha=0.1, loc="lower right")
     plt.tight_layout()
     st.pyplot(fig2, use_container_width=True); plt.close()

 st.markdown("<hr>", unsafe_allow_html=True)

 b1, b2 = st.columns([1.3, 1])

 with b1:
     st.markdown('<p class="sec-label">Confusion Matrix</p>', unsafe_allow_html=True)
     fig3, axes = plt.subplots(1,2,figsize=(8,3))
     for i,(mn,preds) in enumerate([("XGBoost",xgb_m.predict(Xte)),("Logistic Regression",lr_m.predict(Xte_s))]):
         cm = confusion_matrix(yte, preds)
         axes[i].imshow(cm, cmap="Blues")
         axes[i].set_title(mn, fontsize=10, fontweight="600", color="#64748b")
         axes[i].set_xticks([0,1]); axes[i].set_yticks([0,1])
         axes[i].set_xticklabels(["Neg","Pos"], fontsize=8)
         axes[i].set_yticklabels(["Neg","Pos"], fontsize=8)
         axes[i].set_xlabel("Predicted", fontsize=8); axes[i].set_ylabel("Actual", fontsize=8)
         for r in range(2):
             for c in range(2):
                 axes[i].text(c,r,str(cm[r,c]),ha="center",va="center",fontsize=16,fontweight="700",
                     color="white" if cm[r,c]>cm.max()*0.5 else "#1e293b")
     plt.tight_layout(pad=1.5)
     st.pyplot(fig3, use_container_width=True); plt.close()

 with b2:
     st.markdown('<p class="sec-label">Patient Summary</p>', unsafe_allow_html=True)
     rows = [("Age",age),("Sex",sex),("Chest Pain",cp),("Resting BP",f"{trestbps} mmHg"),
             ("Cholesterol",f"{chol} mg/dL"),("Fasting BS",fbs),("ECG",restecg),
             ("Max HR",thalch),("Exercise Angina",exang),("ST Depression",oldpeak),
             ("ST Slope",slope),("Vessels",ca),("Thalassemia",thal)]
     tbl = "".join(f"<tr><td>{k}</td><td style='color:#e2e8f0;font-weight:500;'>{v}</td></tr>" for k,v in rows)
     st.markdown(f'<table class="ptable"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>{tbl}</tbody></table>',
         unsafe_allow_html=True)

else:
 st.markdown('<p class="sec-label">Exploratory Analysis</p>', unsafe_allow_html=True)
 e1, e2 = st.columns(2)
 with e1:
     fig, ax = plt.subplots(figsize=(6.5,3.6))
     cp_g = df_raw.groupby(["cp","target"]).size().unstack(fill_value=0)
     cp_g.plot(kind="bar",ax=ax,color=["#3b82f6","#ef4444"],edgecolor="#0d1424",linewidth=1,width=0.6)
     ax.set_title("Chest Pain × Disease Outcome", fontsize=11, fontweight="600", color="#64748b")
     ax.set_xlabel(""); ax.set_ylabel("Patients", fontsize=9)
     ax.legend(["Healthy","Disease"], fontsize=8, framealpha=0)
     ax.tick_params(axis="x", rotation=15, labelsize=8)
     ax.grid(axis="y", alpha=0.3)
     ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
     plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
 with e2:
     fig, ax = plt.subplots(figsize=(6.5,3.6))
     ax.hist(df_raw[df_raw["target"]==0]["thalch"].dropna(),bins=22,alpha=0.8,color="#3b82f6",label="Healthy",linewidth=0)
     ax.hist(df_raw[df_raw["target"]==1]["thalch"].dropna(),bins=22,alpha=0.8,color="#ef4444",label="Disease",linewidth=0)
     ax.set_title("Max Heart Rate × Disease Outcome", fontsize=11, fontweight="600", color="#64748b")
     ax.set_xlabel("bpm", fontsize=9); ax.set_ylabel("Patients", fontsize=9)
     ax.legend(fontsize=8, framealpha=0); ax.grid(axis="y", alpha=0.3)
     ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
     plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

 st.markdown("""
 <div style='text-align:center;padding:24px 0 8px;color:#1e293b;font-size:13px;'>
     ← &nbsp; Configure patient vitals and click <span style='color:#3b82f6;font-weight:600;'>Predict Cardiac Risk</span>
 </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin-top:40px;'>", unsafe_allow_html=True)
components.html("""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background: transparent; font-family: 'Inter', sans-serif; padding: 0 0 8px 0; }

.footer-card {
 max-width: 560px;
 margin: 0 auto;
 background: linear-gradient(135deg, #0a0f1e 0%, #0d1a38 100%);
 border: 1px solid #1e2d4a;
 border-radius: 20px;
 padding: 28px 32px;
 text-align: center;
}
.built-by {
 font-size: 10px;
 font-weight: 700;
 letter-spacing: 2px;
 text-transform: uppercase;
 color: #2563eb;
 margin-bottom: 8px;
}
.dev-name {
 font-size: 24px;
 font-weight: 800;
 color: #f1f5f9;
 letter-spacing: -0.5px;
 margin-bottom: 4px;
}
.dev-title {
 font-size: 12px;
 color: #334155;
 margin-bottom: 22px;
 font-weight: 400;
}
.links {
 display: flex;
 justify-content: center;
 gap: 10px;
 flex-wrap: wrap;
}
.link-btn {
 display: inline-flex;
 align-items: center;
 gap: 7px;
 padding: 9px 18px;
 border-radius: 9px;
 font-size: 12px;
 font-weight: 500;
 text-decoration: none;
 transition: opacity 0.2s;
}
.link-btn:hover { opacity: 0.82; }
.gh  { background: #0d1117; border: 1px solid #21262d; color: #e6edf3; }
.li  { background: #0a66c2; color: white; }
.ig  { background: linear-gradient(135deg, #833ab4 0%, #e1306c 50%, #f77737 100%); color: white; }
</style>
</head>
<body>
<div class="footer-card">
 <div class="built-by">Crafted by</div>
 <div class="dev-name">Pranshu Kumar</div>
 <div class="dev-title">ML Engineer &nbsp;·&nbsp; Healthcare AI</div>
 <div class="links">
     <a class="link-btn gh" href="https://github.com/dev-pranshu04" target="_blank">
         <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
             <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.38.6.1.82-.26.82-.58v-2.03c-3.34.72-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.74.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.83 2.8 1.3 3.49 1 .1-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.14-.3-.54-1.52.1-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02 0 2.04.13 3 .4 2.28-1.55 3.29-1.23 3.29-1.23.65 1.66.24 2.88.12 3.18.77.84 1.23 1.91 1.23 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.69.83.57C20.57 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z"/>
         </svg>
         GitHub
     </a>
     <a class="link-btn li" href="https://www.linkedin.com/in/dev-pranshu" target="_blank">
         <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
             <path d="M20.45 20.45h-3.55v-5.57c0-1.33-.03-3.04-1.85-3.04-1.85 0-2.13 1.45-2.13 2.94v5.67H9.37V9h3.41v1.56h.05c.47-.9 1.63-1.85 3.36-1.85 3.59 0 4.26 2.36 4.26 5.44v6.3zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.55V9h3.57v11.45zM22.23 0H1.77C.79 0 0 .77 0 1.72v20.56C0 23.23.79 24 1.77 24h20.46c.98 0 1.77-.77 1.77-1.72V1.72C24 .77 23.21 0 22.23 0z"/>
         </svg>
         LinkedIn
     </a>
     <a class="link-btn ig" href="https://www.instagram.com/im_pranshu29/" target="_blank">
         <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
             <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838a6.162 6.162 0 1 0 0 12.324 6.162 6.162 0 0 0 0-12.324zM12 16a4 4 0 1 1 0-8 4 4 0 0 1 0 8zm6.406-11.845a1.44 1.44 0 1 0 0 2.881 1.44 1.44 0 0 0 0-2.881z"/>
         </svg>
         Instagram
     </a>
 </div>
</div>
</body>
</html>
""", height=200)
