"""
CardioRisk AI — PDF Report Generator
======================================
Generates a downloadable clinical-style PDF summary using ReportLab.
Falls back gracefully if reportlab is not installed.
"""

from __future__ import annotations

import io
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False

from src.config import (
    APP_NAME, APP_VERSION, MODEL_VERSION,
    RISK_LABELS, RISK_NOTES,
    RISK_LOW_THRESHOLD, RISK_MED_THRESHOLD,
    FEATURE_LABELS,
)


def generate_pdf_report(
    inputs: dict,
    result,
    bundle,
) -> bytes | None:
    """
    Build a PDF report as bytes. Returns None if reportlab is missing.

    Parameters
    ----------
    inputs  : raw sidebar input dict (label values, not encoded)
    result  : PredictionResult from models.predict()
    bundle  : ModelBundle for version metadata
    """
    if not _HAS_REPORTLAB:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Colour palette ─────────────────────────────────────────────────────
    NAVY   = colors.HexColor("#0d1b35")
    BLUE   = colors.HexColor("#3b82f6")
    GREY   = colors.HexColor("#64748b")
    WHITE  = colors.white
    RED    = colors.HexColor("#ef4444")
    AMBER  = colors.HexColor("#f59e0b")
    GREEN  = colors.HexColor("#10b981")

    p = result.prob_ensemble
    if p < RISK_LOW_THRESHOLD:
        tier, risk_colour = "low",  GREEN
    elif p < RISK_MED_THRESHOLD:
        tier, risk_colour = "med",  AMBER
    else:
        tier, risk_colour = "high", RED

    # ── Title ──────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title", fontSize=22, textColor=NAVY, fontName="Helvetica-Bold",
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub", fontSize=10, textColor=GREY, fontName="Helvetica", spaceAfter=12,
    )
    story.append(Paragraph("🫀 CardioRisk AI — Clinical Risk Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')} · "
        f"App v{APP_VERSION} · Model v{MODEL_VERSION}",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=16))

    # ── Risk Score Box ─────────────────────────────────────────────────────
    score_data = [[
        Paragraph(f"<b>{result.risk_pct:.1f}%</b>", ParagraphStyle(
            "Score", fontSize=36, textColor=risk_colour, fontName="Helvetica-Bold",
        )),
        Paragraph(
            f"<b>{RISK_LABELS[tier]}</b><br/>"
            f"<font size=9 color='#64748b'>{RISK_NOTES[tier]}</font>",
            ParagraphStyle("RiskNote", fontSize=11, textColor=NAVY,
                           fontName="Helvetica-Bold", leading=16),
        ),
    ]]
    score_table = Table(score_data, colWidths=[5*cm, 12*cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX",        (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ("ROWPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Ensemble breakdown ─────────────────────────────────────────────────
    story.append(Paragraph("<b>Model Ensemble Breakdown</b>",
                           ParagraphStyle("Heading2", fontSize=12, textColor=NAVY,
                                          fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6)))
    ensemble_data = [
        ["Model",       "Probability", "Weight"],
        ["XGBoost",     f"{result.prob_xgb*100:.1f}%",      "65%"],
        ["Logistic Reg.", f"{result.prob_lr*100:.1f}%",     "35%"],
        ["Ensemble",    f"{result.prob_ensemble*100:.1f}%", "—"],
    ]
    ens_table = Table(ensemble_data, colWidths=[6*cm, 5*cm, 5*cm])
    ens_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWPADDING",  (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND",  (0, -1), (-1, -1), colors.HexColor("#eff6ff")),
        ("FONTNAME",    (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    story.append(ens_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Patient Inputs ─────────────────────────────────────────────────────
    story.append(Paragraph("<b>Patient Input Summary</b>",
                           ParagraphStyle("Heading2", fontSize=12, textColor=NAVY,
                                          fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6)))
    display_keys = [
        ("age", "Age"),
        ("sex", "Sex"),
        ("cp_label", "Chest Pain Type"),
        ("trestbps", "Resting BP (mmHg)"),
        ("chol", "Cholesterol (mg/dL)"),
        ("thalch", "Max Heart Rate (bpm)"),
        ("oldpeak", "ST Depression"),
        ("ca", "Fluoroscopy Vessels"),
        ("restecg_label", "Resting ECG"),
        ("slope_label", "ST Slope"),
        ("thal_label", "Thalassemia"),
        ("fbs", "Fasting BS > 120"),
        ("exang", "Exercise Angina"),
    ]
    input_rows = [["Parameter", "Value"]]
    for key, label in display_keys:
        val = inputs.get(key, "—")
        if isinstance(val, bool):
            val = "Yes" if val else "No"
        input_rows.append([label, str(val)])

    inp_table = Table(input_rows, colWidths=[9*cm, 8*cm])
    inp_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWPADDING",  (0, 0), (-1, -1), 6),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(inp_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Top Feature Importances ────────────────────────────────────────────
    story.append(Paragraph("<b>Top Feature Importances (XGBoost)</b>",
                           ParagraphStyle("Heading2", fontSize=12, textColor=NAVY,
                                          fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6)))
    fi_items = list(result.feature_importances.items())[:8]
    fi_rows  = [["Feature", "Importance"]] + [
        [feat, f"{val:.4f}"] for feat, val in fi_items
    ]
    fi_table = Table(fi_rows, colWidths=[11*cm, 6*cm])
    fi_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWPADDING",  (0, 0), (-1, -1), 6),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(fi_table)
    story.append(Spacer(1, 0.6*cm))

    # ── Disclaimer ─────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=GREY, spaceAfter=8))
    disc_style = ParagraphStyle(
        "Disc", fontSize=7.5, textColor=GREY, fontName="Helvetica",
        leading=11, spaceAfter=0,
    )
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by CardioRisk AI for research "
        "and educational purposes only. It is not a certified medical device and must "
        "not be used as a substitute for professional clinical judgment. Always consult "
        "a qualified healthcare provider before making any medical decisions.",
        disc_style,
    ))

    doc.build(story)
    return buf.getvalue()
