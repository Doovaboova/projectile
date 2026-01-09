import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from ml.solver import ProjectileSolver

try:
    from ml.generate_dataset import generate_trajectory
except ImportError:
    from ml.generate_data import generate_trajectory

# 🔵 NEW
from llm.interpreter import interpret_results

st.set_page_config(
    page_title="Projectile Motion Inference",
    layout="wide",
    page_icon="🚀"
)

st.sidebar.title("Projectile Motion Inference")
mode = st.sidebar.radio("Select Module:", ["Trajectory Simulator"])

if mode == "Trajectory Simulator":
    st.title("Trajectory Inference Simulator")

    with st.sidebar.form("sim_settings"):
        true_v0 = st.slider("Velocity (m/s)", 10.0, 100.0, 50.0)
        true_angle = st.slider("Launch Angle (°)", 15.0, 85.0, 45.0)
        true_drag = st.slider("Drag Coefficient (k)", 0.005, 0.05, 0.02)
        noise = st.slider("Sensor Noise", 0.1, 5.0, 1.0)
        run_sim = st.form_submit_button("Launch 🚀")

    if run_sim:
        t = np.linspace(0, 8, 40)
        params = [true_v0, true_angle, 0.0, 1.0, true_drag]

        x_clean, y_clean = generate_trajectory(params, t)
        x_noisy = x_clean + np.random.normal(0, noise, len(t))
        y_noisy = np.maximum(y_clean + np.random.normal(0, noise, len(t)), 0)

        solver = ProjectileSolver()
        traj = np.column_stack((t, x_noisy, y_noisy))

        ml_guess = solver.infer(traj)
        refined_guess = solver.refine(ml_guess, t, x_noisy, y_noisy)
        ref_x, ref_y = solver.simulate_physics(refined_guess, t)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_noisy, y=y_noisy, mode="markers", name="Sensor"))
        fig.add_trace(go.Scatter(x=x_clean, y=y_clean, mode="lines", name="True"))
        fig.add_trace(go.Scatter(x=ref_x, y=ref_y, mode="lines", name="AI"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 AI Scientific Interpretation")

        with st.spinner("Analyzing results..."):
            explanation = interpret_results(
                true_params={
                    "v0": true_v0,
                    "angle": true_angle,
                    "k": true_drag
                },
                inferred_params=refined_guess,
                noise=noise
            )

        st.markdown(explanation)
