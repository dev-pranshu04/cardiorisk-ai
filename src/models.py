"""
CardioRisk AI — Model Layer
============================
Trains Logistic Regression + XGBoost ensemble, exposes a single
`predict()` function, and stores version metadata alongside results.

Key fixes applied
-----------------
P0-3 : @st.cache_resource guard is applied at call-site in app.py;
        train_models() itself is a plain cached function — the double
        if/else session_state anti-pattern is removed.
P2-9 : CV is done with Pipeline(scaler + model) to prevent leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    ENSEMBLE_WEIGHTS,
    LR_PARAMS,
    MODEL_VERSION,
    RANDOM_SEED,
    TEST_SIZE,
    XGB_PARAMS,
    FEATURE_COLS,
    FEATURE_LABELS,
)

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ModelBundle:
    """Everything produced by train_models() — passed around as one object."""
    lr:         LogisticRegression
    xgb:        XGBClassifier
    scaler:     StandardScaler
    lr_auc:     float
    xgb_auc:    float
    cv_auc:     float
    lr_ap:      float
    xgb_ap:     float
    cv_ap:      float
    X_test:     np.ndarray
    y_test:     np.ndarray
    X_test_s:   np.ndarray       # scaled test set
    version:    str  = MODEL_VERSION
    trained:    str  = field(default_factory=lambda: str(date.today()))
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))


@dataclass
class PredictionResult:
    """Output of a single predict() call."""
    prob_lr:       float    # raw LR probability
    prob_xgb:      float    # raw XGB probability
    prob_ensemble: float    # weighted blend
    label:         int      # 0 or 1
    risk_pct:      float    # ensemble * 100
    feature_importances: dict[str, float]   # xgb feature importances (named)
    input_df:      pd.DataFrame             # 1-row input for SHAP / logging


# ── Training ──────────────────────────────────────────────────────────────────

def train_models(X: np.ndarray, y: np.ndarray) -> ModelBundle:
    """
    Split, scale, fit LR + XGB, evaluate, return ModelBundle.
    Decorated with @st.cache_resource at call-site in app.py.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # ── Logistic Regression ──────────────────────────────────────────────────
    pos_ratio = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_tr_s, y_tr)
    lr_prob   = lr.predict_proba(X_te_s)[:, 1]
    lr_auc    = roc_auc_score(y_te, lr_prob)
    lr_ap     = average_precision_score(y_te, lr_prob)

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb = XGBClassifier(scale_pos_weight=pos_ratio, **XGB_PARAMS)
    xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    xgb_prob  = xgb.predict_proba(X_te)[:, 1]
    xgb_auc   = roc_auc_score(y_te, xgb_prob)
    xgb_ap    = average_precision_score(y_te, xgb_prob)

    # ── Cross-validation (P2-9: pipeline prevents leakage) ───────────────────
    cv_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb",    XGBClassifier(scale_pos_weight=pos_ratio, **XGB_PARAMS)),
    ])
    skf       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(cv_pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    cv_ap_scores = cross_val_score(cv_pipe, X, y, cv=skf, scoring="average_precision", n_jobs=-1)
    cv_auc = float(cv_scores.mean())
    cv_ap  = float(cv_ap_scores.mean())

    logger.info(
        "Training complete | LR AUC=%.4f | XGB AUC=%.4f | CV AUC=%.4f±%.4f",
        lr_auc, xgb_auc, cv_auc, cv_scores.std(),
    )

    return ModelBundle(
        lr=lr, xgb=xgb, scaler=scaler,
        lr_auc=lr_auc, xgb_auc=xgb_auc, cv_auc=cv_auc,
        lr_ap=lr_ap, xgb_ap=xgb_ap, cv_ap=cv_ap,
        X_test=X_te, y_test=y_te, X_test_s=X_te_s,
    )


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(bundle: ModelBundle, input_df: pd.DataFrame) -> PredictionResult:
    """
    Run ensemble prediction on a 1-row input DataFrame.
    Returns PredictionResult with probabilities, label, and named importances.
    """
    X_raw   = input_df[FEATURE_COLS].values.astype(np.float32)
    X_scaled = bundle.scaler.transform(X_raw)

    p_lr  = float(bundle.lr.predict_proba(X_scaled)[0, 1])
    p_xgb = float(bundle.xgb.predict_proba(X_raw)[0, 1])

    w_xgb = ENSEMBLE_WEIGHTS["xgb"]
    w_lr  = ENSEMBLE_WEIGHTS["lr"]
    p_ens = w_xgb * p_xgb + w_lr * p_lr

    label = int(p_ens >= 0.5)

    # Named XGBoost feature importances
    imp_raw  = bundle.xgb.feature_importances_
    fi = {
        FEATURE_LABELS.get(col, col): float(val)
        for col, val in zip(FEATURE_COLS, imp_raw)
    }
    # Sort descending
    fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    return PredictionResult(
        prob_lr=p_lr,
        prob_xgb=p_xgb,
        prob_ensemble=p_ens,
        label=label,
        risk_pct=p_ens * 100,
        feature_importances=fi,
        input_df=input_df.copy(),
    )
