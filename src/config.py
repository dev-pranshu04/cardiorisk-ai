"""
Central configuration: file paths, feature lists, labels, color palette,
matplotlib theme, and model metadata.
"""
import base64
from pathlib import Path
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT_DIR / "heart_disease_uci.csv"
LOGO_PATH   = ROOT_DIR / "assets" / "Cardiorisk_logo.png"

# ─────────────────────────────────────────────────────────────────────────────
# LOGO (loaded by reference, not embedded — P1-5)
# ─────────────────────────────────────────────────────────────────────────────
def get_logo_uri() -> str:
    """Return a data: URI for the logo, or an empty string if missing."""
    try:
        data = LOGO_PATH.read_bytes()
        b64 = base64.b64encode(data).decode()
        return f"data:image/png;base64,{b64}"
    except FileNotFoundError:
        return ""


LOGO_URI = get_logo_uri()

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
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

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
def apply_matplotlib_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG2, "axes.facecolor": BG2,
        "axes.edgecolor": GRID, "axes.labelcolor": T1,
        "xtick.color": T0, "ytick.color": T0,
        "text.color": T2, "grid.color": GRID, "grid.linewidth": 0.4,
        "font.family": "DejaVu Sans", "font.size": 9,
    })


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────
FEATS = [
    "age", "sex_enc", "cp_enc", "trestbps", "chol", "fbs_enc", "restecg_enc",
    "thalch", "exang_enc", "oldpeak", "slope_enc", "ca", "thal_enc",
]

FLABELS = {
    "age":         "Age",
    "sex_enc":     "Biological Sex",
    "cp_enc":      "Chest Pain Type",
    "trestbps":    "Resting BP",
    "chol":        "Cholesterol",
    "fbs_enc":     "Fasting Blood Sugar",
    "restecg_enc": "Resting ECG",
    "thalch":      "Max Heart Rate",
    "exang_enc":   "Exercise Angina",
    "oldpeak":     "ST Depression",
    "slope_enc":   "ST Slope",
    "ca":          "Major Vessels",
    "thal_enc":    "Thalassemia",
}

# ─────────────────────────────────────────────────────────────────────────────
# ENCODING MAPS (shared between data.py and predict.py)
# ─────────────────────────────────────────────────────────────────────────────
CP_MAP      = {"typical angina": 0, "atypical angina": 1, "non-anginal": 2, "asymptomatic": 3}
CP_MAP_UI   = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal": 2, "Asymptomatic": 3}
RESTECG_MAP = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
RESTECG_MAP_UI = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
SLOPE_MAP   = {"upsloping": 0, "flat": 1, "downsloping": 2}
SLOPE_MAP_UI = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
THAL_MAP    = {"normal": 1, "fixed defect": 2, "reversable defect": 3}
THAL_MAP_UI = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}
BOOL_MAP    = {True: 1, False: 0, "True": 1, "False": 0}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL / ENSEMBLE METADATA
# ─────────────────────────────────────────────────────────────────────────────
MODEL_VERSION = "1.1.0"
XGB_WEIGHT = 0.65
LR_WEIGHT  = 0.35

XGB_PARAMS = dict(
    n_estimators=150, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, verbosity=0,
    tree_method="hist", n_jobs=-1,
)
