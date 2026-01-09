import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time

from ml.solver import ProjectileSolver

# Robust import for generator
try:
    from ml.generate_dataset import generate_trajectory
except ImportError:
    from ml.generate_data import generate_trajectory

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Projectile Motion Inference",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
st.sidebar.title("Projectile Motion Inference")
st.sidebar.caption("Physics AI Engine v1.0")

# Navigation Menu
mode = st.sidebar.radio(
    "Select Module:",
    ["Trajectory Simulator", "System Validation Lab"]
)

st.sidebar.divider()

# =========================================
# MODULE 1: TRAJECTORY SIMULATOR (The App)
# =========================================
if mode == "Trajectory Simulator":
    st.title("Trajectory Inference Simulator")
    st.markdown("""
    **Mission:** Launch a projectile and let the AI deduce the physics parameters from the flight path.
    """)

    # 1. CONTROLS (Form)
    with st.sidebar.form("sim_settings"):
        st.header("Launch Configuration")
        true_v0 = st.slider("Velocity (m/s)", 10.0, 100.0, 50.0)
        true_angle = st.slider("Launch Angle (°)", 15.0, 85.0, 45.0)
        true_drag = st.slider("Drag Coefficient (k)", 0.005, 0.05, 0.02, format="%.3f")
        noise = st.slider("Sensor Noise Level", 0.1, 5.0, 1.0)

        run_sim = st.form_submit_button("Launch Projectile 🚀", use_container_width=True)

    # 2. SIMULATION LOGIC
    if run_sim:
        # Generate Data
        t = np.linspace(0, 8.0, 40)
        params = [true_v0, true_angle, 0.0, 1.0, true_drag]
        x_clean, y_clean = generate_trajectory(params, t)

        if x_clean is not None:
            # Add Noise
            x_noisy = x_clean + np.random.normal(0, noise, len(t))
            y_noisy = np.maximum(y_clean + np.random.normal(0, noise, len(t)), 0.0)

            # AI Inference
            solver = ProjectileSolver()

            with st.spinner("Processing Telemetry..."):
                # Initial ML Guess
                ml_guess = solver.infer(np.column_stack((t, x_noisy, y_noisy)))
                # Physics Refinement
                refined_guess = solver.refine(ml_guess, t, x_noisy, y_noisy)

                # Simulations for plotting
                ref_x, ref_y = solver.simulate_physics(refined_guess, t)

            # 3. VISUALIZATION
            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader("Real-time Flight Tracking")
                fig = go.Figure()

                # The Sensor Data
                fig.add_trace(go.Scatter(
                    x=x_noisy, y=y_noisy,
                    mode='markers',
                    name='Sensor Data',
                    marker=dict(color='#888', opacity=0.5, size=8)
                ))

                # The Actual Path
                fig.add_trace(go.Scatter(
                    x=x_clean, y=y_clean,
                    mode='lines',
                    name='Actual Path',
                    line=dict(color='green', dash='dash', width=2)
                ))

                # The AI Solution
                fig.add_trace(go.Scatter(
                    x=ref_x, y=ref_y,
                    mode='lines',
                    name='AI Solution',
                    line=dict(color='#FF4B4B', width=4)
                ))

                fig.update_layout(
                    xaxis_title="Distance (m)",
                    yaxis_title="Altitude (m)",
                    template="plotly_dark",
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Inference Results")

                # Custom Metric Cards
                def metric_card(label, true, pred, unit=""):
                    delta = pred - true
                    st.metric(
                        label=label,
                        value=f"{pred:.2f}{unit}",
                        delta=f"{delta:.2f} error",
                        delta_color="inverse"
                    )
                    st.divider()

                metric_card("Velocity", true_v0, refined_guess['v0'], " m/s")
                metric_card("Angle", true_angle, refined_guess['angle'], "°")
                metric_card("Drag Coeff", true_drag, refined_guess['k'])

# =========================================
# MODULE 2: SYSTEM VALIDATION LAB (Stress Testing)
# =========================================
elif mode == "System Validation Lab":
    st.title("System Validation Lab")
    st.markdown("Automated stress testing to verify model reliability under varying conditions.")

    solver = ProjectileSolver()

    tab1, tab2 = st.tabs(["🧪 Noise Robustness", "📉 Data Sparsity"])

    # --- TAB 1: NOISE TEST ---
    with tab1:
        st.info("Objective: Determine the 'Break Point' where sensor noise overwhelms the AI.")

        if st.button("Run Noise Diagnostics"):
            results = []
            noise_range = np.linspace(0.1, 4.0, 8)
            progress_bar = st.progress(0)

            for i, n in enumerate(noise_range):
                errs = []
                # Run 8 fast trials per noise level
                for _ in range(8):
                    # Random Physics
                    v = np.random.uniform(30, 80)
                    params = [v, 45, 0, 1, 0.02]
                    t = np.linspace(0, 8, 40)
                    xc, yc = generate_trajectory(params, t)

                    if xc is not None:
                        # Add Noise
                        traj = np.column_stack((
                            t,
                            xc + np.random.normal(0, n, 40),
                            yc + np.random.normal(0, n, 40)
                        ))
                        # Infer
                        pred = solver.infer(traj)
                        errs.append(abs(pred['v0'] - v))

                if errs:
                    results.append({"Noise Level": n, "Velocity Error (MAE)": np.mean(errs)})

                progress_bar.progress((i + 1) / len(noise_range))

            # Plot Results
            df = pd.DataFrame(results)
            st.markdown("### 📊 Diagnostic Results")
            st.line_chart(df, x="Noise Level", y="Velocity Error (MAE)", color="#FF4B4B")

            st.success("Diagnostics Complete. If the curve is linear, the model is robust.")

    # --- TAB 2: SPARSITY TEST ---
    with tab2:
        st.info("Objective: Determine how little data the model needs to make an accurate prediction.")

        if st.button("Run Sparsity Diagnostics"):
            results_time = []
            durations = [2.0, 4.0, 6.0, 8.0] # Observation window
            progress_bar = st.progress(0)

            for i, dur in enumerate(durations):
                errs = []
                for _ in range(8):
                    v = 50
                    t = np.linspace(0, dur, 30) # Only observe 'dur' seconds
                    params = [v, 45, 0, 1, 0.02]
                    xc, yc = generate_trajectory(params, t)
                    if xc is not None:
                        traj = np.column_stack((t, xc + np.random.normal(0,0.5,30), yc + np.random.normal(0,0.5,30)))
                        pred = solver.infer(traj)
                        errs.append(abs(pred['v0'] - v))

                if errs:
                    results_time.append({"Observation Time (s)": dur, "Error": np.mean(errs)})
                progress_bar.progress((i+1)/len(durations))

            df_time = pd.DataFrame(results_time)
            st.markdown("### 📊 Diagnostic Results")
            st.line_chart(df_time, x="Observation Time (s)", y="Error", color="#00CC96")
