# [CardioRisk AI — Project Case Study](https://cardiorisk-ai-pk04.streamlit.app/#cardio-risk-ai)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![AUC](https://img.shields.io/badge/AUC--ROC-0.915-brightgreen)

**Pranshu Kumar · ML Engineer · Healthcare AI**
[GitHub](https://github.com/dev-pranshu04) · [LinkedIn](https://www.linkedin.com/in/dev-pranshu) 

---

## Project Overview

**CardioRisk AI** is a production-grade clinical decision-support tool that predicts cardiovascular disease risk from 13 patient vitals and ECG features. It combines an XGBoost gradient-boosting model with a calibrated Logistic Regression baseline in a weighted ensemble, deployed as an interactive Streamlit web application with a custom dark medical-grade UI.

| Attribute | Detail |
|---|---|
| **Domain** | Healthcare AI / Clinical Decision Support |
| **Task** | Binary classification — cardiovascular disease risk |
| **Dataset** | UCI Heart Disease (multi-site: Cleveland, Hungary, Switzerland, VA Long Beach) |
| **Patients** | 920 across 4 clinical institutions |
| **Features** | 13 clinical, demographic, and ECG variables |
| **Models** | XGBoost + Logistic Regression (weighted ensemble) |
| **Best AUC-ROC** | **~0.91** (XGBoost) |
| **5-Fold CV AUC** | **~0.90** (stable, low variance) |
| **Stack** | Python · Streamlit · XGBoost · scikit-learn · Pandas · Matplotlib |

---

## Problem Statement

Cardiovascular disease is the leading cause of death globally. Early-stage risk stratification using routine clinical measurements can guide timely referral and intervention — but busy clinicians need a fast, explainable tool that surfaces risk from structured EHR data without requiring imaging or invasive procedures.

**Goal:** Build an end-to-end ML pipeline that ingests 13 standard clinical features, outputs a calibrated disease-probability score, and presents actionable risk tiers with full model diagnostics.

---

## Dataset

The UCI Heart Disease dataset aggregates records from four clinical sites, providing geographic and demographic diversity that mitigates single-centre bias.

| Site | Patients |
|---|---|
| Cleveland Clinic Foundation | 303 |
| Hungarian Institute of Cardiology | 294 |
| University Hospital Zurich (Switzerland) | 123 |
| VA Medical Center, Long Beach | 200 |

**Target variable:** Binary label derived from `num` field — `0` (no disease) vs `1` (disease present, `num ≥ 1`). The dataset shows approximately 55% disease prevalence post-binarisation.

**Features used:**

| Feature | Type | Clinical Meaning |
|---|---|---|
| `age` | Numeric | Patient age in years |
| `sex` | Binary | Biological sex (Male=1) |
| `cp` | Ordinal | Chest pain type (0–3) |
| `trestbps` | Numeric | Resting blood pressure (mmHg) |
| `chol` | Numeric | Serum cholesterol (mg/dL) |
| `fbs` | Binary | Fasting blood sugar >120 mg/dL |
| `restecg` | Ordinal | Resting ECG result |
| `thalch` | Numeric | Maximum heart rate achieved |
| `exang` | Binary | Exercise-induced angina |
| `oldpeak` | Numeric | ST depression (exercise vs rest) |
| `slope` | Ordinal | Slope of peak exercise ST segment |
| `ca` | Numeric | Major vessels coloured by fluoroscopy (0–3) |
| `thal` | Ordinal | Thalassemia type |

---

## ML Pipeline Architecture

```
Raw CSV
   │
   ▼
Encoding  ──────────────────────────────────────────────
   │  • Ordinal mapping (cp, restecg, slope, thal)      │
   │  • Binary flags (sex, fbs, exang)                  │
   │                                                    │
   ▼                                                    │
Imputation (Median strategy, sklearn SimpleImputer)    │
   │                                                    │
   ├──────────────────────┬────────────────────────────┘
   │                      │
   ▼                      ▼
StandardScaler       Raw features
(for LR only)        (XGBoost is
   │                 scale-invariant)
   ▼                      │
Logistic        XGBoost Classifier
Regression      (n_est=300, md=4,
(C=0.5,          lr=0.04, subsample
 L2 reg)         =0.8, scale_pos_weight)
   │                      │
   └──────────┬───────────┘
              ▼
     Weighted Ensemble
     (LR×0.35 + XGB×0.65)
              │
              ▼
     Risk Score + Tier
     (Low / Moderate / High)
```

---

## Modelling Decisions — The "Why"

### Why XGBoost as Primary?

XGBoost was chosen because:
- Cardiology features contain **non-linear interactions** (e.g. ST depression × slope × vessel count) that tree-based models handle natively.
- It is **robust to missing values** (internal sparsity-aware split-finding), critical for a dataset with ~8% missingness.
- `scale_pos_weight` provides built-in class imbalance correction without requiring SMOTE, avoiding synthetic sample artefacts in medical data.
- Extensive hyperparameter tuning (early stopping, regularisation via `gamma`, `reg_alpha`, `reg_lambda`) was used to prevent overfitting on a moderately-sized dataset.

### Why Logistic Regression as Secondary?

- Acts as a **calibrated probabilistic baseline** — LR naturally outputs well-calibrated probabilities on linearly separable feature subspaces.
- Provides **interpretability backstop**: if a clinician questions the XGBoost output, LR coefficients offer a clear linear explanation.
- Ensembling with LR smooths the XGBoost output on edge cases, reducing overconfident high/low predictions.

### Why a Weighted Ensemble (65/35)?

- The 65% XGBoost weight reflects its superior discriminative performance (higher AUC, AP).
- The 35% LR weight retains calibration benefits and reduces variance on out-of-distribution patients.
- Weights were selected by grid-searching the ensemble combination on a held-out validation fold, optimising AUC-ROC.

### Missing Data Strategy

Median imputation was chosen over:
- **Mean** — susceptible to outliers (cholesterol > 500 mg/dL exists in dataset).
- **Model-based (MissForest)** — adds complexity and training-time dependency; unjustified on <10% missingness.
- **Dropping rows** — would disproportionately remove Swiss/VA records (higher missingness) and bias site representation.

---

## Model Performance

| Metric | XGBoost | Logistic Regression | Ensemble |
|---|---|---|---|
| AUC-ROC (test) | **~0.910** | ~0.880 | **~0.905** |
| Average Precision | **~0.915** | ~0.885 | ~0.908 |
| 5-Fold CV AUC | **~0.900** | — | — |
| F1-Score (Disease) | ~0.85 | ~0.82 | ~0.85 |

*Exact values vary slightly per run due to random splits; the app displays live computed metrics.*

**Key findings:**
- `ca` (fluoroscopy vessels) and `thal` (thalassemia type) are consistently the top two features by XGBoost gain — both are well-established clinical risk markers.
- `oldpeak` (ST depression) ranks third, consistent with cardiology literature.
- Asymptomatic chest pain type has the strongest association with positive outcome — counterintuitive but medically validated (silent ischaemia).

---

## Feature Importance Insights

```
Major Vessels (ca)       ████████████████ 0.18  ← Top predictor
Thalassemia (thal)       ██████████████   0.15
ST Depression (oldpeak)  ████████████     0.12
Max Heart Rate (thalch)  ██████████       0.10
Chest Pain Type (cp)     █████████        0.09
Age                      ████████         0.08
ST Slope (slope)         ███████          0.07
...
```

Features above the 9% threshold are flagged in the app's importance chart.

---

## System Architecture & Engineering

### Why Streamlit?
- Enables **single-file full-stack deployment** — no FastAPI backend, no React frontend required.
- Native Python integration means the ML pipeline and UI live in the same process with zero serialisation overhead.
- `st.cache_data` / `st.cache_resource` decorators ensure the model is trained **once per session** — training overhead is incurred only on cold start.

### Caching Strategy
```python
@st.cache_data       # ← Serialises CSV + feature engineering
def load_data(): ...

@st.cache_resource   # ← Caches model objects (not serialisable as data)
def train_models(): ...
```

### UI Architecture
- Custom CSS injected via `st.markdown` — bypasses Streamlit's opinionated component styling.
- `components.html` for the footer card — enables full DOM control outside Streamlit's render tree.
- Matplotlib global `rcParams` override creates a cohesive dark-theme chart system without per-plot styling.

---

## Diagnostics Provided (Post-Prediction)

| Panel | What it Shows |
|---|---|
| **ROC Curve** | True vs false positive trade-off for both models |
| **Precision–Recall Curve** | Performance under class imbalance (more informative than ROC for skewed data) |
| **Calibration Curve** | Whether predicted probabilities match empirical outcomes (critical in medicine) |
| **Confusion Matrix** | Raw TP/FP/TN/FN counts on the held-out test set |
| **Threshold Sensitivity** | How precision, recall, and F1 shift across decision thresholds |
| **Classification Report** | Per-class precision, recall, F1, and support |
| **Feature Importance** | XGBoost gain-based importance with high-signal flags |

---

## Risk Stratification Logic

```python
if ensemble_probability < 0.30:
    tier = "LOW RISK"
    action = "Routine monitoring. Maintain healthy lifestyle."

elif ensemble_probability < 0.60:
    tier = "MODERATE RISK"
    action = "Further evaluation — stress test, echocardiogram, lipid panel."

else:
    tier = "HIGH RISK"
    action = "Cardiology referral indicated. Prompt clinical evaluation."
```

Thresholds (0.30, 0.60) were selected to:
- Minimise false negatives in the HIGH tier (patient safety priority).
- Maintain clinical actionability — MODERATE is a meaningful "investigate further" signal, not a vague midpoint.

---

## How to Run

```bash
# 1. Clone / copy project files
git clone https://github.com/dev-pranshu04/cardiorisk-ai
cd cardiorisk-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure dataset is present
#    Place heart_disease_uci.csv in the project root

# 4. Launch
streamlit run app.py
```

**Environment:** Python 3.10+, tested on macOS and Ubuntu 22.04.

---

## Project Structure

```
cardiorisk-ai/
├── app.py                    # Full application (UI + ML pipeline)
├── requirements.txt          # Pinned dependencies
├── heart_disease_uci.csv     # UCI dataset (4 sites merged)
└── CASE_STUDY.md             # This file
```
---

## Skills Demonstrated

- **Machine Learning:** Classification, ensemble methods, hyperparameter tuning, cross-validation, class imbalance handling, model calibration
- **ML Engineering:** Scikit-learn pipelines, XGBoost, feature engineering, missing data strategies, train/test discipline
- **Data Analysis:** EDA, multi-site dataset handling, feature importance interpretation, clinical domain translation
- **Software Engineering:** Streamlit app architecture, caching strategy, modular code, CSS/HTML custom UI
- **Healthcare AI:** Risk stratification logic, calibration importance in clinical ML, explainability considerations, responsible AI disclaimers
- **Visualisation:** ROC, PR curves, calibration plots, confusion matrix, threshold analysis, feature importance charts

---

*CardioRisk AI — Built by Pranshu Kumar. Research and educational use only. Not a certified medical device.*
