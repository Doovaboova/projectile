import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os

# ==========================================
# 🔧 CONFIGURATION (Final Production Grade)
# ==========================================
NUM_SAMPLES = 10000
TIME_STEPS = 50           # Input size for CNN
DURATION = 8.0            # Seconds
NOISE_STD = 0.5           # Sensor noise (meters)
OUTPUT_DIR = "projectile_dataset"
GRAVITY = 9.81

# Parameter Ranges
RANGES = {
    'v0': [10.0, 100.0],
    'angle': [15.0, 75.0],
    'x0': [-5.0, 5.0],
    'y0': [1.0, 10.0],    # Min height 1.0m
    'k': [0.005, 0.05]    # Drag coeff (will be log-sampled)
}

# ==========================================
# 📐 PHYSICS ENGINE
# ==========================================
def projectile_ode(t, state, k):
    """
    Differential equations with quadratic drag.
    State: [x, y, vx, vy]
    """
    x, y, vx, vy = state
    v = np.sqrt(vx**2 + vy**2)

    ax = -k * v * vx
    ay = -GRAVITY - (k * v * vy)

    return [vx, vy, ax, ay]

def hit_ground(t, state):
    """
    Event listener: Stop simulation when y crosses 0.
    """
    return state[1]

hit_ground.terminal = True
hit_ground.direction = -1

def generate_trajectory(params, t_target):
    """
    Simulates trajectory with precise ground collision,
    interpolation, and padding.
    """
    v0, angle, x0, y0, k = params

    angle_rad = np.radians(angle)
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)

    # Solve ODE until impact or max duration
    sol = solve_ivp(
        fun=lambda t, y: projectile_ode(t, y, k),
        t_span=(0, t_target[-1] + 1.0), # Add buffer to ensure we cover full duration if needed
        y0=[x0, y0, vx, vy],
        events=hit_ground,
        max_step=0.1,
        method='RK45'
    )

    # Fail if integration failed or flight was too short (< 4 points)
    if not sol.success or len(sol.t) < 4:
        return None, None

    # --- INTERPOLATION & PADDING ---

    # 1. Create interpolator (No extrapolation to prevent artifacts)
    f_x = interp1d(sol.t, sol.y[0], kind='cubic', bounds_error=False, fill_value=np.nan)
    f_y = interp1d(sol.t, sol.y[1], kind='cubic', bounds_error=False, fill_value=np.nan)

    x_out = np.zeros_like(t_target)
    y_out = np.zeros_like(t_target)

    flight_duration = sol.t[-1]

    # Mask for flight vs ground
    mask_flight = t_target <= flight_duration
    mask_ground = ~mask_flight

    # Interpolate flight portion
    if np.any(mask_flight):
        x_out[mask_flight] = f_x(t_target[mask_flight])
        y_out[mask_flight] = f_y(t_target[mask_flight])

    # Pad ground portion (Ball sits at final position)
    if np.any(mask_ground):
        x_out[mask_ground] = sol.y[0][-1] # Final x
        y_out[mask_ground] = 0.0          # Final y

    return x_out, y_out

# ==========================================
# 🚀 MAIN GENERATION LOOP
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🚀 Generating {NUM_SAMPLES} PRODUCTION-GRADE samples...")
    print(f"   - Log-uniform drag sampling")
    print(f"   - Clamped ground noise")
    print(f"   - Safe cubic interpolation")

    X_data = []
    y_labels = []
    metadata = []

    t_eval = np.linspace(0, DURATION, TIME_STEPS)

    count = 0
    while count < NUM_SAMPLES:
        if count % 1000 == 0:
            print(f"   ... generated {count} samples")

        # 1. Sample Parameters
        v0 = np.random.uniform(*RANGES['v0'])
        angle = np.random.uniform(*RANGES['angle'])
        x0 = np.random.uniform(*RANGES['x0'])
        y0 = np.random.uniform(*RANGES['y0'])

        # Log-Uniform Sampling for k
        log_k_min = np.log(RANGES['k'][0])
        log_k_max = np.log(RANGES['k'][1])
        k = np.exp(np.random.uniform(log_k_min, log_k_max))

        params = [v0, angle, x0, y0, k]

        # 2. Simulate
        x_clean, y_clean = generate_trajectory(params, t_eval)

        if x_clean is None:
            continue

        # 3. Add Noise & CLAMP
        x_noisy = x_clean + np.random.normal(0, NOISE_STD, size=TIME_STEPS)
        y_noisy_raw = y_clean + np.random.normal(0, NOISE_STD, size=TIME_STEPS)

        # Clamp y >= 0 (Ground is hard)
        y_noisy = np.maximum(y_noisy_raw, 0.0)

        # 4. Store
        trajectory_matrix = np.column_stack((t_eval, x_noisy, y_noisy))

        X_data.append(trajectory_matrix)
        y_labels.append(params)

        metadata.append({
            'id': count,
            'v0': v0, 'angle': angle, 'x0': x0, 'y0': y0, 'k': k,
            'flight_time': t_eval[y_clean > 0][-1] if np.any(y_clean > 0) else 0
        })

        count += 1

    # Convert & Save
    X_data = np.array(X_data, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.float32)

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_data)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_labels)
    pd.DataFrame(metadata).to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    print("\n✅ Final Dataset Generated!")
    print(f"   Shape X: {X_data.shape}")
    print(f"   Shape y: {y_labels.shape}")

if __name__ == "__main__":
    main()
