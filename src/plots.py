"""
All matplotlib figure-producing functions, rendered via st.pyplot.

Each function wraps figure creation/closing in try/finally (P2-6) so that
an exception mid-plot doesn't leak an open figure.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score,
)
from sklearn.calibration import calibration_curve

from src.config import BG, BG2, GRID, T0, T1, T2, BLUE, RED, GRN, PRP


def _strip_top_right(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def render_eda_overview(df_raw, y):
    """Class distribution / age histogram / patients-by-site (3-panel)."""
    fig = plt.figure(figsize=(10, 4.2), facecolor=BG2)
    try:
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

        ax0 = fig.add_subplot(gs[0])
        vals = y.value_counts().sort_index()
        wedges, texts, autotexts = ax0.pie(
            [vals[0], vals[1]], labels=["Healthy", "Disease"],
            colors=[BLUE, RED], autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 8, "color": T2},
            wedgeprops={"linewidth": 2.5, "edgecolor": BG2},
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_color("#f1f5f9")
            at.set_fontweight("600")
        ax0.set_title("Class Distribution", fontsize=9, fontweight="600", color=T1, pad=10)

        ax1 = fig.add_subplot(gs[1])
        ax1.hist(df_raw[df_raw.target == 0]["age"], bins=20, alpha=0.8, color=BLUE,
                 label="Healthy", linewidth=0, rwidth=0.9)
        ax1.hist(df_raw[df_raw.target == 1]["age"], bins=20, alpha=0.75, color=RED,
                 label="Disease", linewidth=0, rwidth=0.9)
        ax1.set_title("Age by Outcome", fontsize=9, fontweight="600", color=T1, pad=10)
        ax1.set_xlabel("Age (years)", fontsize=8)
        ax1.set_ylabel("Count", fontsize=8)
        ax1.legend(fontsize=7.5, framealpha=0, labelcolor=T2)
        ax1.grid(axis="y", alpha=0.35)
        _strip_top_right(ax1)

        ax2 = fig.add_subplot(gs[2])
        src = df_raw["dataset"].value_counts()
        bar_colors = [BLUE, GRN, "#f59e0b", PRP][:len(src)]
        bars = ax2.bar(src.index, src.values, color=bar_colors, edgecolor=BG2,
                        linewidth=2, width=0.55)
        ax2.set_title("Patients by Site", fontsize=9, fontweight="600", color=T1, pad=10)
        ax2.tick_params(axis="x", rotation=18, labelsize=7)
        ax2.grid(axis="y", alpha=0.35)
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                     str(int(b.get_height())), ha="center", fontsize=8, color=T1)
        _strip_top_right(ax2)

        plt.tight_layout(pad=1.2)
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def render_feature_importance(xgb_model, feats, flabels):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG2)
    try:
        import pandas as pd
        imp_vals = xgb_model.feature_importances_
        fi = pd.DataFrame({
            "feature": [flabels[f] for f in feats],
            "importance": imp_vals,
        }).sort_values("importance", ascending=True)

        bar_cols = [RED if v > 0.09 else BLUE for v in fi["importance"]]
        bars = ax.barh(fi["feature"], fi["importance"],
                        color=bar_cols, edgecolor=BG, linewidth=1, height=0.65)
        ax.axvline(0.09, color=GRID, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Feature Importance (Gain)", fontsize=9, color=T1)
        ax.grid(axis="x", alpha=0.25)
        for b, val in zip(bars, fi["importance"]):
            ax.text(val + 0.003, b.get_y() + b.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8, color=T0)
        _strip_top_right(ax)
        p1 = mpatches.Patch(color=RED, label=">9% — High signal")
        p2 = mpatches.Patch(color=BLUE, label="Standard signal")
        ax.legend(handles=[p1, p2], fontsize=8, framealpha=0.05,
                  loc="lower right", labelcolor=T2)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def render_roc_curve(y_test, xgb_prob, lr_prob):
    fig, ax = plt.subplots(figsize=(4, 3.6), facecolor=BG2)
    try:
        for prob, lbl, col in [(xgb_prob, "XGBoost", BLUE), (lr_prob, "LR", PRP)]:
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            ax.plot(fpr, tpr, color=col, lw=1.6, label=f"{lbl} ({auc:.3f})")
        ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, linestyle="--")
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.set_title("ROC Curve", fontsize=9, fontweight="600", color=T1, pad=8)
        ax.legend(fontsize=7.5, framealpha=0.05, labelcolor=T2)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _strip_top_right(ax)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def render_pr_curve(y_test, xgb_prob, lr_prob):
    fig, ax = plt.subplots(figsize=(4, 3.6), facecolor=BG2)
    try:
        for prob, lbl, col in [(xgb_prob, "XGBoost", BLUE), (lr_prob, "LR", PRP)]:
            p_, r_, _ = precision_recall_curve(y_test, prob)
            ap = average_precision_score(y_test, prob)
            ax.plot(r_, p_, color=col, lw=1.6, label=f"{lbl} AP={ap:.3f}")
        ax.axhline(y_test.mean(), color=GRID, linestyle="--", lw=0.8)
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.set_title("Precision\u2013Recall", fontsize=9, fontweight="600", color=T1, pad=8)
        ax.legend(fontsize=7.5, framealpha=0.05, labelcolor=T2)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _strip_top_right(ax)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def render_calibration_curve(y_test, xgb_prob, lr_prob):
    fig, ax = plt.subplots(figsize=(4, 3.6), facecolor=BG2)
    try:
        for prob, lbl, col in [(xgb_prob, "XGBoost", BLUE), (lr_prob, "LR", PRP)]:
            frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=8)
            ax.plot(mean_pred, frac_pos, marker="o", ms=4, color=col, lw=1.5, label=lbl)
        ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", lw=0.8, label="Perfect")
        ax.set_xlabel("Mean Predicted Probability", fontsize=8)
        ax.set_ylabel("Fraction of Positives", fontsize=8)
        ax.set_title("Calibration Curve", fontsize=9, fontweight="600", color=T1, pad=8)
        ax.legend(fontsize=7.5, framealpha=0.05, labelcolor=T2)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _strip_top_right(ax)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def render_confusion_matrix(y_test, xgb_model, X_test):
    fig, ax = plt.subplots(figsize=(4, 3.6), facecolor=BG2)
    try:
        cm = confusion_matrix(y_test, xgb_model.predict(X_test))
        cmap = LinearSegmentedColormap.from_list("blues", [BG2, BLUE], N=256)
        ax.imshow(cm, cmap=cmap, vmin=0)
        ax.set_title("XGBoost Confusion Matrix", fontsize=9, fontweight="600", color=T1, pad=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Neg", "Predicted Pos"], fontsize=7.5)
        ax.set_yticklabels(["Actual Neg", "Actual Pos"], fontsize=7.5)
        thresh = cm.max() * 0.55
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        fontsize=18, fontweight="700",
                        color="#fff" if cm[r, c] > thresh else T1)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def render_threshold_sensitivity(y_test, xgb_prob_te):
    """Precision/Recall/F1 vs decision threshold (P0-3: imports at module top)."""
    fig, ax = plt.subplots(figsize=(6.5, 2.5), facecolor=BG2)
    try:
        thresholds = np.linspace(0.1, 0.9, 80)
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            preds = (xgb_prob_te >= t).astype(int)
            precisions.append(precision_score(y_test, preds, zero_division=0))
            recalls.append(recall_score(y_test, preds, zero_division=0))
            f1s.append(f1_score(y_test, preds, zero_division=0))

        ax.plot(thresholds, precisions, color=BLUE, lw=1.5, label="Precision")
        ax.plot(thresholds, recalls, color=RED, lw=1.5, label="Recall")
        ax.plot(thresholds, f1s, color=GRN, lw=1.8, label="F1-Score", linestyle="--")
        ax.axvline(0.5, color=T0, linestyle=":", lw=0.8)
        ax.set_xlabel("Decision Threshold", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.legend(fontsize=7.5, framealpha=0.05, labelcolor=T2, ncol=3)
        ax.grid(alpha=0.2)
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(0, 1.05)
        _strip_top_right(ax)
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def render_chest_pain_outcome(df_raw):
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG2)
    try:
        cp_g = df_raw.groupby(["cp", "target"]).size().unstack(fill_value=0)
        x_pos = np.arange(len(cp_g))
        w = 0.38
        if 0 in cp_g.columns:
            ax.bar(x_pos - w / 2, cp_g[0], width=w, color=BLUE, edgecolor=BG,
                   linewidth=1.5, label="Healthy", alpha=0.9)
        if 1 in cp_g.columns:
            ax.bar(x_pos + w / 2, cp_g[1], width=w, color=RED, edgecolor=BG,
                   linewidth=1.5, label="Disease", alpha=0.9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cp_g.index, rotation=14, fontsize=8)
        ax.set_title("Chest Pain Type vs. Disease Outcome", fontsize=10, fontweight="600", color=T1, pad=10)
        ax.set_ylabel("Patient Count", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.05, labelcolor=T2)
        ax.grid(axis="y", alpha=0.3)
        _strip_top_right(ax)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def render_max_hr_outcome(df_raw):
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG2)
    try:
        ax.hist(df_raw[df_raw.target == 0]["thalch"].dropna(), bins=24, alpha=0.8,
                color=BLUE, label="Healthy", linewidth=0, rwidth=0.88)
        ax.hist(df_raw[df_raw.target == 1]["thalch"].dropna(), bins=24, alpha=0.75,
                color=RED, label="Disease", linewidth=0, rwidth=0.88)
        ax.set_title("Maximum Heart Rate vs. Disease Outcome", fontsize=10, fontweight="600", color=T1, pad=10)
        ax.set_xlabel("Max Heart Rate (bpm)", fontsize=9)
        ax.set_ylabel("Patient Count", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.05, labelcolor=T2)
        ax.grid(axis="y", alpha=0.3)
        _strip_top_right(ax)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)
