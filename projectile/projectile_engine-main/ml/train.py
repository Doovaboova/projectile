import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import joblib

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
# Robustly find paths relative to this script's location
# This ensures it works no matter where you run the command from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # .../projectile_app/ml
DATASET_DIR = os.path.join(BASE_DIR, "projectile_dataset")

# We save the model artifacts in the same folder as the script (ml/)
# so they are kept together with the engine code.
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "projectile_model.pth")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "scalers.pkl")

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🧠 MODEL ARCHITECTURE (User Specified CNN)
# ==========================================
class ParameterInferenceNet(nn.Module):
    def __init__(self):
        super(ParameterInferenceNet, self).__init__()

        # Temporal Encoder
        # Input: (Batch, 3, Sequence_Length)
        self.encoder = nn.Sequential(
            # Layer 1: Capture immediate temporal dynamics
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            # Layer 2: Capture complex patterns
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # Regressor: Maps features to 5 physical parameters
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5) # Output: [v0, angle, x0, y0, k]
        )

    def forward(self, x):
        # x input shape: (Batch, Seq_Len, 3)
        # PyTorch Conv1d needs: (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)

        # 1. Extract Features
        z = self.encoder(x) # -> (Batch, 128, Seq_Len)

        # 2. Global Average Pooling (Compress time dimension)
        z = z.mean(dim=-1)  # -> (Batch, 128)

        # 3. Regress Parameters
        return self.regressor(z)

# ==========================================
# 📊 DATA LOADING & PROCESSING
# ==========================================
class ProjectileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data():
    print(f"📂 Looking for data in: {DATASET_DIR}")

    X_path = os.path.join(DATASET_DIR, "X_train.npy")
    y_path = os.path.join(DATASET_DIR, "y_train.npy")

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Could not find {X_path}. Did you run generate_dataset.py?")

    # 1. Load Raw Data
    X_raw = np.load(X_path) # (N, 50, 3)
    y_raw = np.load(y_path) # (N, 5)

    # 2. Compute Normalization Stats (Standardization for Inputs)
    X_mean = X_raw.mean(axis=(0, 1))
    X_std = X_raw.std(axis=(0, 1)) + 1e-6

    # 3. Compute Min/Max for Targets (Scaling to 0-1)
    y_min = y_raw.min(axis=0)
    y_max = y_raw.max(axis=0)

    # 4. Apply Normalization
    X_norm = (X_raw - X_mean) / X_std
    y_norm = (y_raw - y_min) / (y_max - y_min)

    print("✅ Data Loaded & Normalized")

    # Save scalers so app.py can use them later
    scalers = {
        'X_mean': X_mean, 'X_std': X_std,
        'y_min': y_min, 'y_max': y_max
    }
    joblib.dump(scalers, SCALER_SAVE_PATH)

    return X_norm, y_norm

# ==========================================
# 🚀 TRAINING LOOP
# ==========================================
def train():
    X, y = load_and_preprocess_data()

    # Split Train/Val
    dataset = ProjectileDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # Init Model
    model = ParameterInferenceNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"🚀 Starting training on {DEVICE}...")

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}]  Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("✅ Training Complete!")
    print(f"🏆 Best Validation Loss: {best_val_loss:.6f}")
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    print(f"💾 Scalers saved to: {SCALER_SAVE_PATH}")

if __name__ == "__main__":
    train()
