"""
CardioRisk AI — Data Layer
===========================
Handles CSV loading, cleaning (P0-2 zero-value fix), encoding,
missingness flags (P1-4), and input validation (P1-3).
All functions are pure; caching is applied at the call-site in app.py.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.config import (
    DATA_FILE,
    ENCODE_CP, ENCODE_EXANG, ENCODE_FBS,
    ENCODE_RESTECG, ENCODE_SLOPE, ENCODE_THAL,
    FEATURE_COLS,
    ZERO_AS_NULL_COLS,
)

logger = logging.getLogger(__name__)


# ── Public: load & prepare training data ──────────────────────────────────────

def load_and_prepare() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    """
    Load heart_disease_uci.csv, clean, encode, impute, and return
    (processed_df, X, y, data_sha256).

    Returns
    -------
    df      : cleaned + encoded DataFrame (for EDA / charts)
    X       : float32 feature matrix (920 × 13)
    y       : binary label array (0 = no disease, 1 = disease)
    sha256  : hex digest of the raw CSV bytes
    """
    csv_path = _find_csv()
    raw_bytes = csv_path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()[:12]

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows × %d cols from %s", *df.shape, csv_path.name)

    df = _clean(df)
    df = _encode(df)
    df = _add_missingness_flags(df)      # P1-4: flag before imputing
    df = _impute(df)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = (df["num"] > 0).astype(int).values
    return df, X, y, sha256


# ── Public: encode a single user input dict ───────────────────────────────────

def encode_input(
    age: int,
    sex: str,           # "Male" | "Female"
    cp: str,            # sidebar display label
    trestbps: int,
    chol: int,
    fbs: bool,
    restecg: str,
    thalch: int,
    exang: bool,
    oldpeak: float,
    slope: str,
    ca: int,
    thal: str,
) -> pd.DataFrame:
    """
    Convert sidebar widget values → a 1-row DataFrame with FEATURE_COLS.
    Raises ValueError on out-of-range inputs (P1-3).
    """
    _validate_ranges(age, trestbps, chol, thalch, oldpeak, ca)

    row = {
        "age":         float(age),
        "sex_enc":     1.0 if sex == "Male" else 0.0,
        "cp_enc":      float(ENCODE_CP[cp.lower()]),
        "trestbps":    float(trestbps),
        "chol":        float(chol),
        "fbs_enc":     float(ENCODE_FBS[fbs]),
        "restecg_enc": float(ENCODE_RESTECG[restecg.lower()]),
        "thalch":      float(thalch),
        "exang_enc":   float(ENCODE_EXANG[exang]),
        "oldpeak":     float(oldpeak),
        "slope_enc":   float(ENCODE_SLOPE[slope.lower()]),
        "ca":          float(ca),
        "thal_enc":    float(ENCODE_THAL[thal.lower()]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


# ── Private helpers ───────────────────────────────────────────────────────────

def _find_csv() -> Path:
    """Locate CSV relative to project root or CWD."""
    candidates = [
        Path(DATA_FILE),
        Path(__file__).parent.parent / DATA_FILE,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find {DATA_FILE!r}. "
        "Ensure it is in the project root or same directory as app.py."
    )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """P0-2: replace physiologically impossible zero values with NaN."""
    for col in ZERO_AS_NULL_COLS:
        if col in df.columns:
            n = (df[col] == 0).sum()
            if n:
                logger.warning("Replacing %d zero-value sentinel(s) in '%s' with NaN", n, col)
            df[col] = df[col].replace(0, np.nan)
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns to integers."""
    df = df.copy()
    df["sex_enc"]     = (df["sex"] == "Male").astype(float)
    df["cp_enc"]      = df["cp"].map(ENCODE_CP).astype(float)
    df["fbs_enc"]     = df["fbs"].map(ENCODE_FBS).astype(float)
    df["restecg_enc"] = df["restecg"].map(ENCODE_RESTECG).astype(float)
    df["exang_enc"]   = df["exang"].map(ENCODE_EXANG).astype(float)   # P0-1 fix applied
    df["slope_enc"]   = df["slope"].map(ENCODE_SLOPE).astype(float)
    df["thal_enc"]    = df["thal"].map(ENCODE_THAL).astype(float)
    return df


def _add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """P1-4: add binary flags before imputation so the model can learn from missingness."""
    high_missing = ["ca", "thal_enc", "slope_enc"]
    for col in high_missing:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(float)
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute all FEATURE_COLS in-place."""
    imp = SimpleImputer(strategy="median")
    df[FEATURE_COLS] = imp.fit_transform(df[FEATURE_COLS])
    return df


def _validate_ranges(
    age: int, trestbps: int, chol: int,
    thalch: int, oldpeak: float, ca: int
) -> None:
    """P1-3: raise ValueError with a helpful message on impossible inputs."""
    checks = [
        (age,      1,   120, "Age"),
        (trestbps, 60,  250, "Resting BP"),
        (chol,     50,  700, "Cholesterol"),
        (thalch,   40,  250, "Max Heart Rate"),
        (oldpeak,  0.0, 10.0,"ST Depression"),
        (ca,       0,   3,   "Fluoroscopy Vessels"),
    ]
    for val, lo, hi, label in checks:
        if not (lo <= val <= hi):
            raise ValueError(
                f"{label} value {val!r} is outside the valid range [{lo}, {hi}]. "
                "Please check the input and try again."
            )
