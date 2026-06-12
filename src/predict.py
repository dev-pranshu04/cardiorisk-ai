"""
Patient input encoding and ensemble risk prediction.
"""
import pandas as pd

from src.config import (
    FEATS, XGB_WEIGHT, LR_WEIGHT,
    CP_MAP_UI, RESTECG_MAP_UI, SLOPE_MAP_UI, THAL_MAP_UI,
)


def encode_input(inputs: dict) -> pd.DataFrame:
    """Convert sidebar form inputs (dict) into a model-ready DataFrame."""
    return pd.DataFrame([[
        inputs["age"],
        1 if inputs["sex"] == "Male" else 0,
        CP_MAP_UI[inputs["cp"]],
        inputs["trestbps"],
        inputs["chol"],
        1 if inputs["fbs"] == "Yes" else 0,
        RESTECG_MAP_UI[inputs["restecg"]],
        inputs["thalch"],
        1 if inputs["exang"] == "Yes" else 0,
        inputs["oldpeak"],
        SLOPE_MAP_UI[inputs["slope"]],
        inputs["ca"],
        THAL_MAP_UI[inputs["thal"]],
    ]], columns=FEATS)


def compute_prediction(inputs: dict, models: dict) -> dict:
    """Run the encoded input through both models and blend the result.

    Returns a dict with individual + ensemble probabilities, risk tier,
    and the original inputs (for the patient summary table / PDF report).
    """
    Xi = encode_input(inputs)
    Xi_imp = pd.DataFrame(models["imputer"].transform(Xi), columns=FEATS)

    xp = models["xgb"].predict_proba(Xi_imp)[0][1]
    lp = models["lr"].predict_proba(models["scaler"].transform(Xi_imp))[0][1]
    ep = xp * XGB_WEIGHT + lp * LR_WEIGHT

    if ep < 0.30:
        risk_cls, risk_tag = "low", "LOW RISK"
        risk_note = "Routine monitoring advised. Continue healthy lifestyle habits and annual review."
    elif ep < 0.60:
        risk_cls, risk_tag = "med", "MODERATE RISK"
        risk_note = "Further evaluation recommended — consider stress test, echocardiogram, and full lipid panel."
    else:
        risk_cls, risk_tag = "high", "HIGH RISK"
        risk_note = "Cardiology referral strongly indicated. Prompt clinical evaluation required."

    return {
        "inputs": inputs,
        "Xi_imp": Xi_imp,
        "xgb_prob": xp,
        "lr_prob": lp,
        "ensemble_prob": ep,
        "risk_cls": risk_cls,
        "risk_tag": risk_tag,
        "risk_note": risk_note,
    }
