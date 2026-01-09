import torch
import numpy as np
import joblib
import os
import sys
from scipy.optimize import minimize

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.train import ParameterInferenceNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "projectile_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scalers.pkl")

class ProjectileSolver:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = None
        self.scalers = None
        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model not found in {BASE_DIR}")

        self.scalers = joblib.load(SCALER_PATH)
        self.model = ParameterInferenceNet()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def infer(self, trajectory):
        """
        Step 1: The Fast Guess (Neural Network)
        Returns: Dictionary of approximate parameters
        """
        # Normalize
        X_mean, X_std = self.scalers['X_mean'], self.scalers['X_std']
        trajectory_norm = (trajectory - X_mean) / X_std

        # Predict
        input_tensor = torch.FloatTensor(trajectory_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_norm = self.model(input_tensor).cpu().numpy()[0]

        # Denormalize
        y_min, y_max = self.scalers['y_min'], self.scalers['y_max']
        pred = pred_norm * (y_max - y_min) + y_min

        return {
            "v0": float(pred[0]),
            "angle": float(pred[1]),
            "x0": float(pred[2]),
            "y0": float(pred[3]),
            "k": float(pred[4])
        }

    def refine(self, initial_params, t_obs, x_obs, y_obs):
        """
        Step 2: The Precision Polish (Physics Optimizer)
        Uses the Neural Network's guess as the starting point (x0) for optimization.
        """

        # 1. Define the Error Function (MSE between Simulation and Data)
        def objective(p):
            # Unpack parameters (v0, angle, k) - we fix x0, y0 to keep it stable
            v, ang, drag = p

            # Constraints (Soft bounds via penalty)
            if v < 0 or v > 150 or ang < 0 or ang > 90 or drag < 0:
                return 1e6

            # Simulate
            pred_x, pred_y = self._simulate_fast(v, ang, initial_params['x0'], initial_params['y0'], drag, t_obs)

            # Calculate Error (MSE)
            error = np.mean((pred_x - x_obs)**2 + (pred_y - y_obs)**2)
            return error

        # 2. Set up the Optimizer
        # Initial Guess from Neural Network
        x0 = [initial_params['v0'], initial_params['angle'], initial_params['k']]

        # Run Optimizer (Nelder-Mead is robust for non-smooth physics)
        res = minimize(objective, x0, method='Nelder-Mead', tol=1e-4)

        # 3. Return Refined Parameters
        return {
            "v0": float(res.x[0]),
            "angle": float(res.x[1]),
            "x0": initial_params['x0'],
            "y0": initial_params['y0'],
            "k": float(res.x[2]),
            "error": float(res.fun)
        }

    def _simulate_fast(self, v0, angle, x0, y0, k, t_eval):
        """Internal simulator for the optimizer (needs to be fast)"""
        vx = v0 * np.cos(np.radians(angle))
        vy = v0 * np.sin(np.radians(angle))
        x, y = x0, y0
        dt = t_eval[1] - t_eval[0]
        path_x, path_y = [], []
        g = 9.81

        for _ in t_eval:
            path_x.append(x)
            path_y.append(y)
            v = np.sqrt(vx**2 + vy**2)

            # Physics Step
            ax = -k * v * vx
            ay = -g - (k * v * vy)

            x += vx * dt
            y += vy * dt
            vx += ax * dt
            vy += ay * dt

            # Simple ground clamp
            if y < 0: y = 0

        return np.array(path_x), np.array(path_y)

    # Wrapper for the App to call
    def simulate_physics(self, params, t_eval):
        return self._simulate_fast(params['v0'], params['angle'], params['x0'], params['y0'], params['k'], t_eval)
