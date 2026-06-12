"""
PDF report generation for a single prediction (P1-7).

Produces a one-page clinical summary: patient inputs, ensemble risk score,
model contribution breakdown, and model performance metadata.
"""
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

from src.config import MODEL_VERSION


def build_pdf_report(result: dict, models: dict) -> bytes:
    """Build a PDF report for a prediction `result` (see predict.compute_prediction)."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=9, textColor=colors.grey,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=6,
    )
    body_style = styles["Normal"]

    story = []
    story.append(Paragraph("CardioRisk AI \u2014 Patient Risk Report", title_style))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} \u00b7 "
        f"Model v{MODEL_VERSION} \u00b7 Research &amp; educational use only",
        subtitle_style,
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cbd5e1")))

    # ── Risk summary ─────────────────────────────────────────────────────
    ep = result["ensemble_prob"]
    risk_color = {
        "low": colors.HexColor("#059669"),
        "med": colors.HexColor("#d97706"),
        "high": colors.HexColor("#dc2626"),
    }[result["risk_cls"]]

    story.append(Paragraph("Risk Assessment", section_style))
    risk_table = Table(
        [[f"{ep*100:.1f}%", result["risk_tag"]]],
        colWidths=[1.6 * inch, 4.4 * inch],
    )
    risk_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (0, 0), 26),
        ("FONTSIZE", (1, 0), (1, 0), 12),
        ("TEXTCOLOR", (0, 0), (-1, -1), risk_color),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(result["risk_note"], body_style))

    # ── Model contributions ─────────────────────────────────────────────
    story.append(Paragraph("Model Contributions", section_style))
    contrib_data = [
        ["Model", "Probability"],
        ["XGBoost (65%)", f"{result['xgb_prob']*100:.1f}%"],
        ["Logistic Regression (35%)", f"{result['lr_prob']*100:.1f}%"],
        ["Ensemble Output", f"{ep*100:.1f}%"],
    ]
    contrib_table = Table(contrib_data, colWidths=[3.5 * inch, 2.5 * inch])
    contrib_table.setStyle(_table_style())
    story.append(contrib_table)

    # ── Patient inputs ───────────────────────────────────────────────────
    story.append(Paragraph("Patient Input Summary", section_style))
    inputs = result["inputs"]
    rows = [
        ["Parameter", "Value"],
        ["Age", f"{inputs['age']} years"],
        ["Sex", inputs["sex"]],
        ["Chest Pain", inputs["cp"]],
        ["Resting BP", f"{inputs['trestbps']} mmHg"],
        ["Cholesterol", f"{inputs['chol']} mg/dL"],
        ["Fasting Blood Sugar > 120", inputs["fbs"]],
        ["Resting ECG", inputs["restecg"]],
        ["Max Heart Rate", f"{inputs['thalch']} bpm"],
        ["Exercise Angina", inputs["exang"]],
        ["ST Depression (oldpeak)", f"{inputs['oldpeak']}"],
        ["ST Slope", inputs["slope"]],
        ["Fluoroscopy Vessels", str(inputs["ca"])],
        ["Thalassemia", inputs["thal"]],
    ]
    patient_table = Table(rows, colWidths=[3.0 * inch, 3.0 * inch])
    patient_table.setStyle(_table_style())
    story.append(patient_table)

    # ── Model metadata ───────────────────────────────────────────────────
    story.append(Paragraph("Model Performance (held-out test set)", section_style))
    perf_rows = [
        ["Metric", "XGBoost", "Logistic Regression"],
        ["AUC-ROC", f"{models['xgb_auc']:.3f}", f"{models['lr_auc']:.3f}"],
        ["Average Precision", f"{models['xgb_ap']:.3f}", f"{models['lr_ap']:.3f}"],
        ["5-fold CV AUC (XGBoost)", f"{models['cv_auc']:.3f}", "\u2014"],
    ]
    perf_table = Table(perf_rows, colWidths=[2.4 * inch, 1.8 * inch, 1.8 * inch])
    perf_table.setStyle(_table_style())
    story.append(perf_table)

    # ── Disclaimer ───────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an automated machine "
        "learning system for research and educational purposes only. It is "
        "not a certified medical device and must not be used for clinical "
        "diagnosis without physician oversight.",
        ParagraphStyle("Disclaimer", parent=body_style, fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def _table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])
