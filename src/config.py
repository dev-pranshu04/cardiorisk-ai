"""
CardioRisk AI — Central Configuration
======================================
Single source of truth for features, encoding maps, model hyperparameters,
theme colours, and version metadata.  Import this module everywhere;
never hard-code these values in app.py or other modules.
"""

from __future__ import annotations

# ── Version & Metadata ────────────────────────────────────────────────────────
APP_NAME        = "CardioRisk AI"
APP_VERSION     = "2.0.0"
MODEL_VERSION   = "2.0.0"
TRAINED_ON_DATA = "UCI Heart Disease (4-site, 920 patients)"
DATA_HASH_NOTE  = "heart_disease_uci.csv — SHA-256 verified at load"

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_FILE = "heart_disease_uci.csv"

# Physiologically impossible sentinel values in UCI data (coded as missing)
ZERO_AS_NULL_COLS = ["trestbps", "chol", "thalch"]

# ── Features ──────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "sex_enc", "cp_enc", "trestbps", "chol",
    "fbs_enc", "restecg_enc", "thalch", "exang_enc",
    "oldpeak", "slope_enc", "ca", "thal_enc",
]

# Human-readable labels (used in charts, tables, tooltips)
FEATURE_LABELS: dict[str, str] = {
    "age":         "Age",
    "sex_enc":     "Biological Sex",
    "cp_enc":      "Chest Pain Type",
    "trestbps":    "Resting BP (mmHg)",
    "chol":        "Cholesterol (mg/dL)",
    "fbs_enc":     "Fasting Blood Sugar",
    "restecg_enc": "Resting ECG",
    "thalch":      "Max Heart Rate (bpm)",
    "exang_enc":   "Exercise Angina",
    "oldpeak":     "ST Depression",
    "slope_enc":   "ST Slope",
    "ca":          "Fluoroscopy Vessels",
    "thal_enc":    "Thalassemia",
}

# Clinical descriptions shown in the "What-If" simulator tooltip
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "age":         "Patient age in years",
    "sex_enc":     "1 = Male, 0 = Female",
    "cp_enc":      "0=Typical Angina, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic",
    "trestbps":    "Resting systolic blood pressure on admission (mmHg)",
    "chol":        "Serum cholesterol in mg/dL",
    "fbs_enc":     "Fasting blood sugar > 120 mg/dL (1=Yes, 0=No)",
    "restecg_enc": "0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy",
    "thalch":      "Maximum heart rate achieved during exercise",
    "exang_enc":   "Exercise-induced angina (1=Yes, 0=No)",
    "oldpeak":     "ST depression induced by exercise relative to rest",
    "slope_enc":   "0=Upsloping, 1=Flat, 2=Downsloping",
    "ca":          "Number of major vessels (0–3) coloured by fluoroscopy",
    "thal_enc":    "1=Normal, 2=Fixed Defect, 3=Reversable Defect",
}

# ── Encoding Maps ─────────────────────────────────────────────────────────────
# Using both Python-native bool and string keys to handle mixed CSV reading
ENCODE_SEX     = {"Male": 1, "Female": 0}
ENCODE_CP      = {
    "typical angina": 0, "atypical angina": 1,
    "non-anginal": 2, "asymptomatic": 3,
}
ENCODE_FBS     = {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
ENCODE_RESTECG = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
ENCODE_EXANG   = {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}  # P0-1 fix
ENCODE_SLOPE   = {"upsloping": 0, "flat": 1, "downsloping": 2}
ENCODE_THAL    = {"normal": 1, "fixed defect": 2, "reversable defect": 3}

# ── Sidebar Input Choices (display → encoded) ─────────────────────────────────
SIDEBAR_CP      = ["Asymptomatic", "Typical Angina", "Atypical Angina", "Non-Anginal"]
SIDEBAR_CP_MAP  = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal": 2, "Asymptomatic": 3}
SIDEBAR_ECG     = ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
SIDEBAR_ECG_MAP = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
SIDEBAR_SLOPE   = ["Upsloping", "Flat", "Downsloping"]
SIDEBAR_SLOPE_MAP = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
SIDEBAR_THAL    = ["Normal", "Fixed Defect", "Reversable Defect"]
SIDEBAR_THAL_MAP = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}

# ── Data Ranges (from actual UCI data — used for slider bounds) ───────────────
AGE_RANGE      = (29, 77)
BP_RANGE       = (90, 200)     # physiologically valid range (data has 0s, ignored)
CHOL_RANGE     = (100, 603)
HR_RANGE       = (60, 202)
OLDPEAK_RANGE  = (0.0, 6.2)
CA_RANGE       = (0, 3)

# ── Model Hyperparameters ─────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators     = 150,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma            = 0.1,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    # eval_metric moved to fit() in XGBoost 3.x
    random_state     = 42,
    verbosity        = 0,
    device           = "cpu",
    n_jobs           = -1,
)

LR_PARAMS = dict(
    max_iter     = 1000,
    C            = 0.5,
    solver       = "lbfgs",
    random_state = 42,
)

# Ensemble weights (must sum to 1.0)
ENSEMBLE_WEIGHTS = {"xgb": 0.65, "lr": 0.35}

CV_FOLDS    = 5
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# ── Risk Thresholds ───────────────────────────────────────────────────────────
RISK_LOW_THRESHOLD  = 0.30   # < 30%  → Low
RISK_MED_THRESHOLD  = 0.60   # 30–60% → Moderate
RISK_HIGH_THRESHOLD = 0.60   # >= 60% → High  (alias for clarity)

RISK_LABELS = {
    "low":  "LOW RISK",
    "med":  "MODERATE RISK",
    "high": "HIGH RISK",
}

RISK_NOTES = {
    "low":  (
        "Routine monitoring advised. Continue healthy lifestyle habits, "
        "maintain target BP and cholesterol, and schedule an annual review."
    ),
    "med":  (
        "Further evaluation recommended — consider stress test, "
        "echocardiogram, and full lipid panel. Follow up with your physician."
    ),
    "high": (
        "Cardiology referral strongly indicated. Prompt clinical evaluation "
        "required. Do not delay assessment."
    ),
}

# ── Plot Theme ────────────────────────────────────────────────────────────────
PLOT_BG   = "#040810"
PLOT_BG2  = "#070d1c"
PLOT_GRID = "#0a1428"
PLOT_T0   = "#334155"
PLOT_T1   = "#475569"
PLOT_T2   = "#64748b"
PLOT_T3   = "#94a3b8"
PLOT_BLUE = "#3b82f6"
PLOT_RED  = "#ef4444"
PLOT_GRN  = "#10b981"
PLOT_PRP  = "#8b5cf6"
PLOT_CYN  = "#06b6d4"
PLOT_ORG  = "#f59e0b"
