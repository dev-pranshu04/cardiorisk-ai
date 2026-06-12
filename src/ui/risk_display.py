"""
Risk score display, model contribution bars, validation warnings,
and PDF download button.
"""
import streamlit as st

from src.config import BLUE, PRP, GRN
from src.pdf_report import build_pdf_report
from src.validation import validate_inputs


def render_validation_warnings(inputs: dict) -> None:
    warnings = validate_inputs(inputs)
    for severity, msg in warnings:
        if severity == "warning":
            st.warning(msg, icon="\u26a0\ufe0f")
        else:
            st.info(msg, icon="\u2139\ufe0f")


def render_risk_score(result: dict) -> None:
    ep = result["ensemble_prob"]
    st.markdown('<p class="sec-hd">Risk Score</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="risk-display {result['risk_cls']}">
        <div class="risk-pct">{ep*100:.1f}%</div>
        <div class="risk-label">{result['risk_tag']}</div>
        <div class="risk-note">{result['risk_note']}</div>
    </div>""", unsafe_allow_html=True)


def render_model_contributions(result: dict) -> None:
    st.markdown('<p class="sec-hd" style="margin-top:4px;">Model Contributions</p>', unsafe_allow_html=True)
    for nm, prob, col in [
        ("XGBoost (65%)", result["xgb_prob"], BLUE),
        ("Logistic Regression (35%)", result["lr_prob"], PRP),
        ("Ensemble Output", result["ensemble_prob"], GRN),
    ]:
        st.markdown(f"""
        <div class="pbar-wrap">
            <div class="pbar-header">
                <span>{nm}</span>
                <span class="val">{prob*100:.1f}%</span>
            </div>
            <div class="pbar-track">
                <div class="pbar-fill" style="width:{prob*100:.1f}%;background:{col};"></div>
            </div>
        </div>""", unsafe_allow_html=True)


def render_pdf_download(result: dict, models: dict) -> None:
    pdf_bytes = build_pdf_report(result, models)
    st.download_button(
        label="\U0001f4c4 Download PDF Report",
        data=pdf_bytes,
        file_name="cardiorisk_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def render_patient_summary_table(inputs: dict) -> None:
    st.markdown('<p class="sec-hd">Patient Input Summary</p>', unsafe_allow_html=True)
    patient_rows = [
        ("Age", f"{inputs['age']} years"), ("Sex", inputs["sex"]), ("Chest Pain", inputs["cp"]),
        ("Resting BP", f"{inputs['trestbps']} mmHg"), ("Cholesterol", f"{inputs['chol']} mg/dL"),
        ("Fasting Blood Sugar", inputs["fbs"]), ("Resting ECG", inputs["restecg"]),
        ("Max Heart Rate", f"{inputs['thalch']} bpm"), ("Exercise Angina", inputs["exang"]),
        ("ST Depression", f"{inputs['oldpeak']}"), ("ST Slope", inputs["slope"]),
        ("Fluoroscopy Vessels", str(inputs["ca"])), ("Thalassemia", inputs["thal"]),
    ]
    tbl = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in patient_rows)
    st.markdown(f"""
    <div class="ptable-wrap">
    <table class="ptable">
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>{tbl}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)


def render_classification_report(y_test, xgb_model, X_test) -> None:
    from sklearn.metrics import classification_report
    st.markdown('<p class="sec-hd">Classification Report \u2014 XGBoost</p>', unsafe_allow_html=True)
    cr = classification_report(y_test, xgb_model.predict(X_test), output_dict=True)
    rows_html = ""
    for cls_name, label in [
        ("0", "Healthy (0)"), ("1", "Disease (1)"),
        ("macro avg", "Macro Avg"), ("weighted avg", "Weighted Avg"),
    ]:
        if cls_name in cr:
            d = cr[cls_name]
            supp = f"{int(d.get('support', 0))}" if "support" in d else "\u2014"
            rows_html += f"""
            <tr>
                <td>{label}</td>
                <td>{d['precision']:.3f}</td>
                <td>{d['recall']:.3f}</td>
                <td>{d['f1-score']:.3f}</td>
                <td>{supp}</td>
            </tr>"""
    st.markdown(f"""
    <div class="ptable-wrap">
    <table class="ptable">
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)
