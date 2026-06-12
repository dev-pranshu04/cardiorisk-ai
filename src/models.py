"""
Model training, cached as a resource.

Fixes:
- P0-2: `train_models` no longer takes unhashable DataFrame arguments.
  It is decorated with @st.cache_resource and internally calls the
  @st.cache_data-decorated `load_data()` to obtain X/y, so Streamlit's
  hashing never sees a DataFrame argument.
- P1-2: a single `cross_validate` call computes both ROC-AUC and average
  precision, instead of two separate `cross_val_score` calls (halves
  cold-start CV time).
- P1-3: XGBoost uses `tree_method="hist"` instead of the deprecated
  `device="cpu"` argument.
"""
import xgboost as xgb
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import XGB_PARAMS
from src.data import load_data


@st.cache_resource
def train_models():
    """Train the LR + XGBoost models and return everything the UI needs.

    Returns
    -------
    dict with keys:
        lr, xgb, scaler, imputer,
        lr_auc, xgb_auc, cv_auc, lr_ap, xgb_ap, cv_ap,
        X_test, y_test, X_test_scaled,
        df_raw, X, y, prevalence, n_patients
    """
    df_raw, X, y, imputer = load_data()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=0.5, solver="lbfgs", random_state=42)
    lr.fit(X_tr_s, y_tr)
    lr_prob = lr.predict_proba(X_te_s)[:, 1]
    lr_auc = roc_auc_score(y_te, lr_prob)
    lr_ap = average_precision_score(y_te, lr_prob)

    # XGBoost — 150 trees is sufficient for this dataset size; keeps cold-start fast
    sp = (y_tr == 0).sum() / (y_tr == 1).sum()
    xgb_m = xgb.XGBClassifier(scale_pos_weight=sp, **XGB_PARAMS)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    xgb_prob = xgb_m.predict_proba(X_te)[:, 1]
    xgb_auc = roc_auc_score(y_te, xgb_prob)
    xgb_ap = average_precision_score(y_te, xgb_prob)

    # Single CV pass with multiple scorers (P1-2)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        xgb_m, X, y, cv=skf,
        scoring={"roc_auc": "roc_auc", "avg_precision": "average_precision"},
        n_jobs=-1,
    )
    cv_auc = cv_results["test_roc_auc"].mean()
    cv_ap = cv_results["test_avg_precision"].mean()

    return {
        "lr": lr,
        "xgb": xgb_m,
        "scaler": sc,
        "imputer": imputer,
        "lr_auc": lr_auc,
        "xgb_auc": xgb_auc,
        "cv_auc": cv_auc,
        "lr_ap": lr_ap,
        "xgb_ap": xgb_ap,
        "cv_ap": cv_ap,
        "X_test": X_te,
        "y_test": y_te,
        "X_test_scaled": X_te_s,
        "df_raw": df_raw,
        "X": X,
        "y": y,
        "prevalence": y.mean(),
        "n_patients": len(df_raw),
    }
