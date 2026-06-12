"""
Sidebar: clinical input panel.
"""
import streamlit as st

from src.config import LOGO_URI


def render_sidebar(df_raw):
    """Render the sidebar inputs and return (inputs_dict, go_clicked)."""
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;padding:12px 0 16px;'>
            <img src='{LOGO_URI}' style='width:64px;height:64px;object-fit:contain;margin-bottom:8px;border-radius:12px;'>
            <div style='font-family:"Playfair Display",serif;font-size:16px;font-weight:700;color:#e2e8f0;'>CardioRisk AI</div>
            <div style='font-size:9px;color:#1e3a5f;margin-top:3px;letter-spacing:2px;text-transform:uppercase;'>Clinical Input Panel</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<p style="color:#1d4ed8;font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:8px 0 6px;">Demographics</p>', unsafe_allow_html=True)
        age = st.slider("Age (years)", int(df_raw.age.min()), int(df_raw.age.max()), 54)
        sex = st.selectbox("Biological Sex", ["Male", "Female"])

        st.markdown('<p style="color:#1d4ed8;font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:8px 0 6px;">Symptoms</p>', unsafe_allow_html=True)
        cp = st.selectbox("Chest Pain Type", ["Asymptomatic", "Typical Angina", "Atypical Angina", "Non-Anginal"])
        exang = st.radio("Exercise-Induced Angina", ["No", "Yes"], horizontal=True)

        st.markdown('<p style="color:#1d4ed8;font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:8px 0 6px;">Vitals & Labs</p>', unsafe_allow_html=True)
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
        chol = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 240)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"], horizontal=True)
        thalch = st.slider("Maximum Heart Rate (bpm)", 70, 210, 150)

        st.markdown('<p style="color:#1d4ed8;font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:8px 0 6px;">ECG & Imaging</p>', unsafe_allow_html=True)
        restecg = st.selectbox("Resting ECG Result", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, 0.1)
        slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        ca = st.slider("Fluoroscopy Vessels (0\u20133)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])

        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("\U0001f50d  Compute Cardiac Risk Score")

    inputs = {
        "age": age, "sex": sex, "cp": cp, "exang": exang,
        "trestbps": trestbps, "chol": chol, "fbs": fbs, "thalch": thalch,
        "restecg": restecg, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    return inputs, go
