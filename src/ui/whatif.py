"""
What-If Simulator (P2-3): adjust key risk factors and instantly see how the
ensemble risk score would change relative to the current prediction.
"""
import streamlit as st

from src.predict import compute_prediction


def render_whatif_simulator(result: dict, models: dict) -> None:
    st.markdown('<p class="sec-hd">What-If Simulator</p>', unsafe_allow_html=True)
    st.caption(
        "Adjust risk factors below to see how the ensemble score would change "
        "for this patient, holding all other inputs constant."
    )

    base_inputs = result["inputs"]

    c1, c2, c3 = st.columns(3)
    with c1:
        sim_chol = st.slider("Cholesterol (mg/dL)", 100, 600, int(base_inputs["chol"]), key="sim_chol")
    with c2:
        sim_trestbps = st.slider("Resting BP (mmHg)", 90, 200, int(base_inputs["trestbps"]), key="sim_bp")
    with c3:
        sim_thalch = st.slider("Max Heart Rate (bpm)", 70, 210, int(base_inputs["thalch"]), key="sim_hr")

    sim_inputs = dict(base_inputs)
    sim_inputs["chol"] = sim_chol
    sim_inputs["trestbps"] = sim_trestbps
    sim_inputs["thalch"] = sim_thalch

    sim_result = compute_prediction(sim_inputs, models)

    base_ep = result["ensemble_prob"] * 100
    sim_ep = sim_result["ensemble_prob"] * 100
    delta = sim_ep - base_ep

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Risk Score", f"{base_ep:.1f}%")
    with col2:
        st.metric("Simulated Risk Score", f"{sim_ep:.1f}%", delta=f"{delta:+.1f} pts", delta_color="inverse")
