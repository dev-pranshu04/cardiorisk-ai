"""
Data loading and feature encoding.

Fixes:
- P0-1: corrected `exang_enc` mapping (was {"False:0": 0}, a silent typo
  that left the string "False" unmapped -> NaN -> imputed wrong).
- P0-4: wraps the CSV read in a try/except with a friendly Streamlit error
  instead of an unhandled FileNotFoundError / raw traceback.
"""
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer

from src.config import (
    DATA_PATH, FEATS,
    CP_MAP, RESTECG_MAP, SLOPE_MAP, THAL_MAP, BOOL_MAP,
)


@st.cache_data
def load_data():
    """Load the heart disease dataset and return (df, X_imputed, y, imputer).

    Halts the app with a clear error message if the dataset file is missing,
    instead of crashing with a raw FileNotFoundError traceback (P0-4).
    """
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(
            "❌ Dataset not found. Ensure `heart_disease_uci.csv` is present "
            "in the project root."
        )
        st.stop()

    df["target"] = (df["num"] > 0).astype(int)
    df["sex_enc"] = (df["sex"] == "Male").astype(int)
    df["cp_enc"] = df["cp"].map(CP_MAP)
    df["fbs_enc"] = df["fbs"].map(BOOL_MAP)
    df["restecg_enc"] = df["restecg"].map(RESTECG_MAP)

    # P0-1 fix: "False" (string) and False (bool) both map to 0; previously
    # the key "False:0" never matched real values, leaving these rows NaN.
    df["exang_enc"] = df["exang"].map(BOOL_MAP)

    df["slope_enc"] = df["slope"].map(SLOPE_MAP)
    df["thal_enc"] = df["thal"].map(THAL_MAP)

    X = df[FEATS]
    y = df["target"]

    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=FEATS)

    return df, X_imp, y, imp
