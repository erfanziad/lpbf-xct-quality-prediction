"""
LPBF In-Situ Monitoring: XCT Voxel Quality Prediction
======================================================
Pipeline:
  1. Load and concatenate per-layer CSV data
  2. Extract melt-pool image features
  3. Run DNN-based melt-pool classification
  4. Build neighbor-augmented (spatial + multi-layer) feature matrix
  5. Train a Transformer regressor to predict XCT pixel grayscale (5×5×5)
  6. Permutation-based feature importance & layer-influence analysis
  7. Detect XCT local extrema and run calibrated anomaly analysis
  8. Visualize results overlaid on XCT slices

Author: Erfan Ziaei-Rad
"""

# ============================================================
# 0) Imports
# ============================================================
import os
import time
import copy
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import shap

from copy import deepcopy
from tqdm import tqdm
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.stats import skew, kurtosis
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ============================================================
# 1) Configuration
# ============================================================

# --- Paths (update to your environment) ---
CSV_DIR = "/path/to/csv_layers"           # Folder with L0001.csv ... L0250.csv
IMAGE_BASE_DIR = "/path/to/tiff_stacks"   # Melt pool TIFF stacks
XCT_VOLUME_PATH = "/path/to/xct_volume"  # 3-D XCT numpy array (.npy)

# --- Column definitions ---
COLUMN_NAMES = [
    "Part number", "Time (µs)", "X position command (mm)", "Y position command (mm)",
    "Laser power command (W)", "Scan speed command (mm/s)", "X position real (mm)",
    "Y position real (mm)", "Laser power real (W)", "Scan speed real (mm/s)",
    "Melt pool length (mm, threshold 80)", "Melt pool width (mm, threshold 80)",
    "Melt pool area (mm², threshold 80)", "Melt pool length (mm, threshold 100)",
    "Melt pool width (mm, threshold 100)", "Melt pool area (mm², threshold 100)",
    "Melt pool length (mm, threshold 120)", "Melt pool width (mm, threshold 120)",
    "Melt pool area (mm², threshold 120)", "Layerwise pixel grayscale (LED a, single)",
    "Layerwise pixel grayscale (LED a, 3x3)", "Layerwise pixel grayscale (LED a, 5x5)",
    "Layerwise pixel grayscale (LED b, single)", "Layerwise pixel grayscale (LED b, 3x3)",
    "Layerwise pixel grayscale (LED b, 5x5)", "Layerwise pixel grayscale (LED c, single)",
    "Layerwise pixel grayscale (LED c, 3x3)", "Layerwise pixel grayscale (LED c, 5x5)",
    "Layerwise pixel grayscale (burned, LED a, single)", "Layerwise pixel grayscale (burned, LED a, 3x3)",
    "Layerwise pixel grayscale (burned, LED a, 5x5)", "Layerwise pixel grayscale (burned, LED b, single)",
    "Layerwise pixel grayscale (burned, LED b, 3x3)", "Layerwise pixel grayscale (burned, LED b, 5x5)",
    "Layerwise pixel grayscale (burned, LED c, single)", "Layerwise pixel grayscale (burned, LED c, 3x3)",
    "Layerwise pixel grayscale (burned, LED c, 5x5)", "XCT pixel grayscale (single)",
    "XCT pixel grayscale (3x3x3)", "XCT pixel grayscale (5x5x5)",
]

# --- Feature/target columns ---
IMAGE_FEATURES = [
    "area", "average_gray_value", "circumference", "convex_hull_area",
    "convex_hull_perimeter", "energy", "entropy", "kurtosis",
    "number_of_regions", "skewness", "smaller_dimension", "std_gray_value",
]
DNN_LABEL_COLUMNS = [
    "Predicted_Category", "Predicted_Spatter", "Predicted_Tail", "Predicted_Plume",
]
FEATURE_COLUMNS = [
    "Laser power real (W)", "Scan speed real (mm/s)",
    "Melt pool length (mm, threshold 80)", "Melt pool width (mm, threshold 80)",
    "Melt pool area (mm², threshold 80)",
    "Layerwise pixel grayscale (LED a, single)", "Layerwise pixel grayscale (LED a, 3x3)",
    "Layerwise pixel grayscale (LED a, 5x5)", "Layerwise pixel grayscale (LED b, single)",
    "Layerwise pixel grayscale (LED b, 3x3)", "Layerwise pixel grayscale (LED b, 5x5)",
    "Layerwise pixel grayscale (LED c, single)", "Layerwise pixel grayscale (LED c, 3x3)",
    "Layerwise pixel grayscale (LED c, 5x5)", "Layerwise pixel grayscale (burned, LED a, single)",
    "Layerwise pixel grayscale (burned, LED a, 3x3)", "Layerwise pixel grayscale (burned, LED a, 5x5)",
    "Layerwise pixel grayscale (burned, LED b, single)", "Layerwise pixel grayscale (burned, LED b, 3x3)",
    "Layerwise pixel grayscale (burned, LED b, 5x5)", "Layerwise pixel grayscale (burned, LED c, single)",
    "Layerwise pixel grayscale (burned, LED c, 3x3)", "Layerwise pixel grayscale (burned, LED c, 5x5)",
] + IMAGE_FEATURES + DNN_LABEL_COLUMNS

TARGET_COLUMN = "XCT pixel grayscale (5x5x5)"

# --- Neighbor / multi-layer config ---
EPS = 0.15           # Spatial neighbor radius (mm)
LAYERS_ABOVE = 5     # Layers above current to include
LAYERS_BELOW = 2     # Layers below current to include

# --- XCT alignment ---
XCT_LAYER_OFFSET = 36   # XCT slice index corresponding to CSV layer 1

# --- Device ---
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2) Data Loading
# ============================================================

def load_csv_layers(csv_dir: str, n_layers: int = 250) -> list[pd.DataFrame]:
    """Load all per-layer CSV files and return as a list of DataFrames."""
    dfs = []
    for i in range(1, n_layers + 1):
        path = os.path.join(csv_dir, f"L{i:04d}.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: {path} not found.")
    print(f"Loaded {len(dfs)} layers.")
    return dfs


def concat_layers(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of per-layer DataFrames into one."""
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ============================================================
# 3) Melt-Pool Image Feature Extraction
# ============================================================

def extract_image_features(image: np.ndarray) -> dict:
    """
    Extract 12 melt-pool morphological and texture features from a grayscale image.
    Returns a dict with default zero values if the image is empty/None.
    """
    default = {k: 0 for k in IMAGE_FEATURES}

    if image is None or np.size(image) == 0:
        return default

    image = np.array(image, dtype=np.float32)
    bright = image[image > 100]
    if bright.size == 0:
        bright = np.array([0.0])

    avg_gray = np.mean(image)
    std_gray = np.std(image)
    skewness = skew(image.flatten()) if image.size > 1 else 0
    kurt = kurtosis(bright.flatten()) if bright.size > 1 else 0
    entropy = -np.sum(image * np.log2(image + 1e-9))

    _, binary = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    n_regions = len(contours)

    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)

    if not regions:
        return {
            **default,
            "average_gray_value": avg_gray, "std_gray_value": std_gray,
            "skewness": skewness, "kurtosis": kurt, "entropy": entropy,
            "number_of_regions": n_regions,
        }

    mp = max(regions, key=lambda r: r.area)
    bbox = mp.bbox
    smaller_dim = min(bbox[2] - bbox[0], bbox[3] - bbox[1])

    hull = convex_hull_image(binary)
    hull_area = np.sum(hull)
    hull_perimeter = measure.perimeter(hull) if np.any(hull) else 0

    try:
        glcm = graycomatrix(image.astype(np.uint8), [1], [0], 256,
                            symmetric=True, normed=True)
        energy = graycoprops(glcm, "energy")[0, 0]
    except Exception:
        energy = 0

    return {
        "area": mp.area, "average_gray_value": avg_gray,
        "circumference": mp.perimeter, "convex_hull_area": hull_area,
        "convex_hull_perimeter": hull_perimeter, "energy": energy,
        "entropy": entropy, "kurtosis": kurt, "number_of_regions": n_regions,
        "skewness": skewness, "smaller_dimension": smaller_dim,
        "std_gray_value": std_gray,
    }


def append_image_features_to_csvs(csv_dir: str, image_base_dir: str,
                                   n_layers: int = 250):
    """
    For each layer, extract melt-pool image features and save them back to the CSV.
    Images are expected at: image_base_dir/Stack_NNN/frame_MMMMM.png
    """
    for i in range(1, n_layers + 1):
        csv_path = os.path.join(csv_dir, f"L{i:04d}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found.")
            continue

        df = pd.read_csv(csv_path)
        if "frame" not in df.columns:
            print(f"Warning: 'frame' column missing in L{i:04d}.csv — skipping.")
            continue

        feature_rows = []
        for frame_num in tqdm(df["frame"].values, desc=f"Layer {i:04d}", leave=False):
            img_path = os.path.join(
                image_base_dir, f"Stack_{i:03d}", f"frame_{int(frame_num):05d}.png"
            )
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_path) else None
            feature_rows.append(extract_image_features(img))

        df = pd.concat([df, pd.DataFrame(feature_rows)], axis=1)
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")


# ============================================================
# 4) DNN-Based Melt-Pool Classification
# ============================================================

class MeltPoolDNN(nn.Module):
    """4-output binary classifier for melt-pool event categories."""

    def __init__(self, input_dim: int = 12, output_dim: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 512),       nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),        nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def run_dnn_inference(csv_dir: str, model_path: str, n_layers: int = 250,
                      device=DEVICE):
    """
    Load a trained MeltPoolDNN, run inference on all CSV layers,
    and append predicted labels back to each CSV.
    """
    model = MeltPoolDNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    print(f"DNN model loaded from {model_path}.")

    for i in range(1, n_layers + 1):
        csv_path = os.path.join(csv_dir, f"L{i:04d}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if not all(c in df.columns for c in IMAGE_FEATURES):
            print(f"Warning: missing image features in L{i:04d}.csv — skipping.")
            continue

        df_clean = df.dropna(subset=IMAGE_FEATURES)
        if df_clean.empty:
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean[IMAGE_FEATURES].values)

        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32).to(device))
            preds = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)

        for j, col in enumerate(DNN_LABEL_COLUMNS):
            df.loc[df_clean.index, col] = preds[:, j]

        df.to_csv(csv_path, index=False)
    print("DNN inference complete.")


# ============================================================
# 5) Neighbor-Augmented Feature Matrix
# ============================================================

def get_spatial_neighbors(df: pd.DataFrame, x0: float, y0: float,
                           eps: float = EPS) -> pd.DataFrame:
    """Return rows within an eps-radius box around (x0, y0), excluding the point itself."""
    return df[
        (df["X position command (mm)"].sub(x0).abs() <= eps) &
        (df["Y position command (mm)"].sub(y0).abs() <= eps) &
        ~((df["X position command (mm)"] == x0) & (df["Y position command (mm)"] == y0))
    ]


def build_multilayer_features(dfs: list[pd.DataFrame], layer_idx: int,
                               eps: float = EPS,
                               layers_above: int = LAYERS_ABOVE,
                               layers_below: int = LAYERS_BELOW
                               ) -> tuple[np.ndarray, np.ndarray]:
    """
    For each point in the current layer, concatenate:
      - its own features
      - mean of spatial neighbors in the current layer
      - mean of spatial neighbors in `layers_above` layers above
      - mean of spatial neighbors in `layers_below` layers below
    Returns (X, y) arrays. Points with missing adjacent layers are skipped.
    """
    X, y = [], []
    df_cur = dfs[layer_idx]

    for _, row in df_cur.iterrows():
        if pd.isna(row[TARGET_COLUMN]):
            continue

        cur_feat = row[FEATURE_COLUMNS].values.astype(float)
        x0 = row["X position command (mm)"]
        y0 = row["Y position command (mm)"]

        all_feat = [cur_feat]

        # Spatial neighbors in the current layer
        nbrs = get_spatial_neighbors(df_cur, x0, y0, eps)
        all_feat.append(nbrs[FEATURE_COLUMNS].mean().values if not nbrs.empty
                        else np.zeros_like(cur_feat))

        # Layers above
        for i in range(1, layers_above + 1):
            idx = layer_idx - i
            if not (0 <= idx < len(dfs)):
                return np.array([]), np.array([])
            df_above = dfs[idx]
            nbrs_above = get_spatial_neighbors(df_above, x0, y0, eps)
            all_feat.append(nbrs_above[FEATURE_COLUMNS].mean().values
                            if not nbrs_above.empty else np.zeros_like(cur_feat))

        # Layers below
        for i in range(1, layers_below + 1):
            idx = layer_idx + i
            if not (0 <= idx < len(dfs)):
                return np.array([]), np.array([])
            df_below = dfs[idx]
            nbrs_below = get_spatial_neighbors(df_below, x0, y0, eps)
            all_feat.append(nbrs_below[FEATURE_COLUMNS].mean().values
                            if not nbrs_below.empty else np.zeros_like(cur_feat))

        X.append(np.concatenate(all_feat))
        y.append(row[TARGET_COLUMN])

    return np.array(X), np.array(y)


def build_feature_names(layers_above: int = LAYERS_ABOVE,
                         layers_below: int = LAYERS_BELOW) -> list[str]:
    names = [f"current_{f}" for f in FEATURE_COLUMNS]
    names += [f"current_layer_neighbors_{f}" for f in FEATURE_COLUMNS]
    for i in range(1, layers_above + 1):
        names += [f"layer_{i}_above_{f}" for f in FEATURE_COLUMNS]
    for i in range(1, layers_below + 1):
        names += [f"layer_{i}_below_{f}" for f in FEATURE_COLUMNS]
    return names


def prepare_dataset(dfs: list[pd.DataFrame],
                    save_path: str = "multilayer_features.csv") -> tuple[np.ndarray, np.ndarray]:
    """Build the full neighbor-augmented dataset across all layers and save to CSV."""
    X_all, y_all = [], []
    for idx in range(len(dfs)):
        print(f"Building features: layer {idx + 1}/{len(dfs)}", end="\r")
        df_layer = dfs[idx].dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
        X_layer, y_layer = build_multilayer_features(dfs, idx)
        if len(X_layer) > 0:
            valid = ~np.isnan(X_layer).any(axis=1) & ~np.isnan(y_layer)
            X_all.append(X_layer[valid])
            y_all.append(y_layer[valid])

    X = np.vstack(X_all) if X_all else np.array([])
    y = np.concatenate(y_all) if y_all else np.array([])

    if len(X) > 0:
        df_out = pd.DataFrame(X, columns=build_feature_names())
        df_out[TARGET_COLUMN] = y
        df_out.to_csv(save_path, index=False)
        print(f"\nSaved feature matrix to {save_path}  ({X.shape})")

    return X, y


# ============================================================
# 6) Transformer Regressor
# ============================================================

class TransformerRegressor(nn.Module):
    """Flat-feature Transformer encoder + regression head."""

    def __init__(self, input_dim: int, num_heads: int = 8, num_layers: int = 2,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, D)
        x = self.transformer(x).squeeze(1)   # (B, D)
        return self.output_head(x).squeeze(-1)


def remove_outliers_iqr(y: np.ndarray, threshold: float = 1.5
                         ) -> tuple[np.ndarray, np.ndarray]:
    q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
    iqr = q3 - q1
    mask = (y < q1 - threshold * iqr) | (y > q3 + threshold * iqr)
    return y[~mask], np.where(mask)[0]


def train_transformer(X: np.ndarray, y: np.ndarray,
                       epochs: int = 200, patience: int = 20,
                       batch_size: int = 64, lr: float = 1e-3,
                       save_path: str = "transformer_model.pth",
                       device=DEVICE
                       ) -> tuple[nn.Module, StandardScaler, MinMaxScaler]:
    """
    Train the TransformerRegressor with early stopping (70/15/15 split).
    Saves a checkpoint and returns the model, feature scaler, and target scaler.
    """
    # Remove outliers and scale target
    y_clean, outlier_idx = remove_outliers_iqr(y)
    X_clean = np.delete(X, outlier_idx, axis=0)
    print(f"Removed {len(outlier_idx)} outliers ({len(outlier_idx)/len(y)*100:.1f}%)")

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y_clean.reshape(-1, 1)).flatten()

    # Train / val / test split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_clean, y_scaled, test_size=0.30, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

    feat_scaler = StandardScaler()
    X_tr_s  = feat_scaler.fit_transform(X_tr)
    X_val_s = feat_scaler.transform(X_val)
    X_te_s  = feat_scaler.transform(X_te)

    def make_loader(Xs, ys, shuffle=False):
        return DataLoader(
            TensorDataset(torch.FloatTensor(Xs), torch.FloatTensor(ys).reshape(-1, 1)),
            batch_size=batch_size, shuffle=shuffle,
        )

    train_loader = make_loader(X_tr_s, y_tr, shuffle=True)
    val_loader   = make_loader(X_val_s, y_val)
    test_loader  = make_loader(X_te_s, y_te)

    model = TransformerRegressor(input_dim=X_tr_s.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb.squeeze(-1)).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += criterion(model(Xb), yb.squeeze(-1)).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | val loss: {val_loss:.5f}")

        if val_loss + 1e-6 < best_val:
            best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    y_pred = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            y_pred.append(model(Xb.to(device)).cpu().numpy())
    y_pred = np.concatenate(y_pred)

    mse = mean_squared_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    print(f"\nTest  MSE: {mse:.5f}  |  R²: {r2:.5f}")

    torch.save({"model_state_dict": model.state_dict(),
                "scaler": feat_scaler, "y_scaler": y_scaler}, save_path)
    print(f"Checkpoint saved to {save_path}")

    return model, feat_scaler, y_scaler


# ============================================================
# 7) Permutation Feature Importance & Layer-Influence Analysis
# ============================================================

def compute_permutation_importance(model: nn.Module, X_test: np.ndarray,
                                    y_test: np.ndarray, feature_names: list[str],
                                    device=DEVICE, n_repeats: int = 3) -> pd.DataFrame:
    """
    Estimate feature importance by measuring the MSE increase after
    randomly permuting each feature column.
    """
    model.eval()
    criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X_test).to(device)
    y_tensor = torch.FloatTensor(y_test).to(device)

    with torch.no_grad():
        baseline_mse = criterion(model(X_tensor), y_tensor).item()

    importances = []
    for col in tqdm(range(X_test.shape[1]), desc="Permutation importance"):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, col])
            with torch.no_grad():
                mse = criterion(
                    model(torch.FloatTensor(X_perm).to(device)), y_tensor
                ).item()
            scores.append(mse - baseline_mse)
        importances.append(np.mean(scores))

    return pd.DataFrame({"feature": feature_names, "importance": importances
                          }).sort_values("importance", ascending=False).reset_index(drop=True)


def compute_layer_influence(importance_df: pd.DataFrame,
                             layers_above: int = LAYERS_ABOVE,
                             layers_below: int = LAYERS_BELOW) -> pd.DataFrame:
    """Aggregate permutation importance by temporal layer group."""
    groups = {"current": 0.0, "current_layer_neighbors": 0.0}
    for i in range(1, layers_above + 1):
        groups[f"layer_{i}_above"] = 0.0
    for i in range(1, layers_below + 1):
        groups[f"layer_{i}_below"] = 0.0

    for _, row in importance_df.iterrows():
        for grp in groups:
            if row["feature"].startswith(grp):
                groups[grp] += row["importance"]

    return pd.DataFrame({"layer": list(groups), "cumulative_importance": list(groups.values())
                          }).sort_values("cumulative_importance", ascending=False).reset_index(drop=True)


# ============================================================
# 8) XCT Extrema Detection
# ============================================================

def detect_xct_extrema(xct_volume: np.ndarray, xct_layer_offset: int = XCT_LAYER_OFFSET,
                        window_size: int = 20, threshold_rel: float = 0.3,
                        neighborhood_radius: int = 15) -> tuple[dict, dict, dict]:
    """
    For each XCT slice, detect local maxima and build a BallTree for fast lookup.
    Returns (coords_dict, tree_dict, region_dict) keyed by XCT layer index.
    """
    coords_dict, tree_dict, region_dict = {}, {}, {}

    for layer_idx in range(xct_volume.shape[0]):
        image = np.flipud(xct_volume[layer_idx])
        smoothed = gaussian_filter(image, sigma=2)
        coords = peak_local_max(smoothed, min_distance=window_size,
                                 threshold_rel=threshold_rel, exclude_border=False)

        key = layer_idx + xct_layer_offset
        coords_dict[key] = coords
        tree_dict[key]   = BallTree(coords) if len(coords) > 0 else None
        region_dict[key] = (coords, neighborhood_radius)

    return coords_dict, tree_dict, region_dict


def is_near_extrema(x_pix: int, y_pix: int, layer: int,
                     tree_dict: dict, region_dict: dict) -> tuple[int, float]:
    """Return (1, confidence) if (x_pix, y_pix) is within the extrema radius, else (0, 0.0)."""
    if layer not in tree_dict or tree_dict[layer] is None:
        return 0, 0.0
    coords, radius = region_dict[layer]
    dist, _ = tree_dict[layer].query([[y_pix, x_pix]], k=1)  # note: row=y, col=x
    if dist[0][0] <= radius:
        return 1, float(1 - dist[0][0] / radius)
    return 0, 0.0


# ============================================================
# 9) Calibrated Anomaly Analysis
# ============================================================

def analyze_layer_predictions(xct_layer: int, csv_idx: int,
                               dfs: list[pd.DataFrame], xct_volume: np.ndarray,
                               model: nn.Module, feat_scaler: StandardScaler,
                               device=DEVICE) -> pd.DataFrame | None:
    """
    For a single XCT layer, compute actual vs. predicted XCT grayscale for
    all non-extrema data points and return a DataFrame with error columns.
    """
    try:
        image = np.flipud(xct_volume[xct_layer])
        smoothed = gaussian_filter(image, sigma=1)
        extrema_coords = set(
            (int(c[1]), int(c[0]))
            for c in peak_local_max(smoothed, min_distance=30, threshold_rel=0.5)
        )

        df_cur = dfs[csv_idx].dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
        results = []

        for _, row in df_cur.iterrows():
            x0 = row["X position command (mm)"]
            y0 = row["Y position command (mm)"]
            cur_feat = row[FEATURE_COLUMNS].values.astype(float)

            # Build neighbor features for this point
            nbrs = get_spatial_neighbors(df_cur, x0, y0)
            nbr_feat = nbrs[FEATURE_COLUMNS].mean().values if not nbrs.empty \
                else np.zeros_like(cur_feat)
            combined = feat_scaler.transform(
                np.concatenate([cur_feat, nbr_feat]).reshape(1, -1)
            )

            with torch.no_grad():
                pred = model(torch.FloatTensor(combined).to(device)).item()

            results.append({
                "x_pix": x0, "y_pix": y0,
                "actual": row[TARGET_COLUMN], "predicted": pred,
            })

        if not results:
            return None

        df_res = pd.DataFrame(results)
        df_res["error"]     = df_res["actual"] - df_res["predicted"]
        df_res["abs_error"] = df_res["error"].abs()
        return df_res

    except Exception as e:
        print(f"Error in layer {xct_layer}: {e}")
        return None


def run_anomaly_analysis(xct_to_csv: dict, dfs: list[pd.DataFrame],
                          xct_volume: np.ndarray, model: nn.Module,
                          feat_scaler: StandardScaler, device=DEVICE,
                          sigma_threshold: float = 3.0) -> pd.DataFrame:
    """
    Run layer-by-layer prediction, compute bias calibration, apply 3-sigma
    anomaly classification, and return a combined results DataFrame.
    """
    all_results = []
    errors_raw  = []

    print("Pass 1: collecting predictions for bias estimation...")
    for xct_layer, csv_file in tqdm(xct_to_csv.items()):
        csv_idx = int(csv_file[1:5]) - 1
        df_res = analyze_layer_predictions(
            xct_layer, csv_idx, dfs, xct_volume, model, feat_scaler, device
        )
        if df_res is not None:
            errors_raw.extend(df_res["error"].dropna().tolist())
            df_res["xct_layer"] = xct_layer
            df_res["csv_file"]  = csv_file
            all_results.append(df_res)

    if not errors_raw:
        raise ValueError("No valid predictions collected.")

    errors_raw  = np.array(errors_raw)
    mean_bias   = np.mean(errors_raw)
    error_std   = np.std(errors_raw - mean_bias)
    threshold   = sigma_threshold * error_std
    print(f"\nMean bias: {mean_bias:.5f}  |  3-sigma threshold: {threshold:.5f}")

    # Calibrate and classify
    final_df = pd.concat(all_results, ignore_index=True)
    final_df["calibrated_prediction"] = final_df["predicted"] + mean_bias
    final_df["calibrated_error"]      = final_df["actual"] - final_df["calibrated_prediction"]
    final_df["abs_calibrated_error"]  = final_df["calibrated_error"].abs()

    final_df["anomaly_type"] = "well_predicted"
    final_df.loc[final_df["calibrated_error"] >  threshold, "anomaly_type"] = "under_predicted"
    final_df.loc[final_df["calibrated_error"] < -threshold, "anomaly_type"] = "over_predicted"

    print("\nAnomaly classification:")
    counts = final_df["anomaly_type"].value_counts()
    print(counts)
    print((counts / len(final_df) * 100).round(2))

    return final_df


# ============================================================
# 10) Visualization
# ============================================================

def plot_xct_error_overlay(results: pd.DataFrame, xct_volume: np.ndarray,
                            xct_layer_offset: int = XCT_LAYER_OFFSET,
                            figsize: tuple = (16, 8)):
    """
    For each XCT layer in results, display:
      - Left: raw XCT slice with detected local maxima
      - Right: prediction error color overlay
    """
    sigma = results["calibrated_error"].std()
    thresholds = {1: sigma, 2: 2 * sigma, 3: 3 * sigma}

    color_map = {
        "strong_under":   (1, 0,   0,   0.9),
        "moderate_under": (1, 0.5, 0,   0.9),
        "mild_under":     (1, 1,   0,   0.9),
        "well_predicted": (0, 1,   0,   0.9),
        "mild_over":      (0, 1,   1,   0.9),
        "moderate_over":  (0, 0,   1,   0.9),
        "strong_over":    (0.5, 0, 0.5, 0.9),
    }

    def classify_error(err):
        ae = abs(err)
        if ae > thresholds[3]:
            return "strong_under" if err > 0 else "strong_over"
        if ae > thresholds[2]:
            return "moderate_under" if err > 0 else "moderate_over"
        if ae > thresholds[1]:
            return "mild_under" if err > 0 else "mild_over"
        return "well_predicted"

    legend = [mpatches.Patch(color=c[:3], label=k)
              for k, c in color_map.items()]

    for layer in sorted(results["xct_layer"].unique()):
        slice_idx = layer - 1 + xct_layer_offset
        if slice_idx >= xct_volume.shape[0]:
            continue

        image = np.flipud(xct_volume[slice_idx])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"XCT Layer {layer}  (slice {slice_idx})", fontsize=14, y=1.02)

        # Left: raw XCT with extrema bounding boxes
        ax1.imshow(image, cmap="gray", origin="lower")
        ax1.set_title("XCT slice with local maxima")
        smoothed = gaussian_filter(image, sigma=1)
        for y, x in peak_local_max(smoothed, min_distance=30, threshold_rel=0.5):
            ax1.add_patch(plt.Rectangle((x - 5, y - 5), 10, 10,
                                        edgecolor="lime", facecolor="none", lw=1))

        # Right: error overlay
        ax2.imshow(image, cmap="gray", origin="lower")
        ax2.set_title("Prediction error classification")
        overlay = np.zeros((*image.shape, 4))
        for _, row in results[results["xct_layer"] == layer].iterrows():
            try:
                x, y = int(round(row["x_pix"])), int(round(row["y_pix"]))
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    overlay[y, x] = color_map[classify_error(row["calibrated_error"])]
            except Exception:
                pass
        ax2.imshow(overlay, origin="lower", interpolation="nearest")

        fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
        plt.tight_layout()
        plt.show()


# ============================================================
# 11) Main Pipeline
# ============================================================

if __name__ == "__main__":

    # --- Step 1: Load per-layer CSVs ---
    dfs = load_csv_layers(CSV_DIR)

    # --- Step 2: Extract melt-pool image features (run once, updates CSVs in-place) ---
    # append_image_features_to_csvs(CSV_DIR, IMAGE_BASE_DIR)
    # dfs = load_csv_layers(CSV_DIR)  # reload after feature extraction

    # --- Step 3: DNN melt-pool classification (run once, updates CSVs in-place) ---
    # run_dnn_inference(CSV_DIR, model_path="feature_based_dnn.pth")
    # dfs = load_csv_layers(CSV_DIR)  # reload after DNN labels

    # --- Step 4: Build neighbor-augmented feature matrix ---
    X, y = prepare_dataset(dfs, save_path="multilayer_features.csv")

    # --- Step 5: Train Transformer regressor ---
    model, feat_scaler, y_scaler = train_transformer(
        X, y, epochs=200, patience=20, save_path="transformer_model.pth"
    )

    # --- Step 6: Feature importance ---
    feature_names = build_feature_names()
    # (Requires X_test_scaled and y_test from the training split; reuse if available)
    # importance_df = compute_permutation_importance(model, X_test_scaled, y_test, feature_names)
    # layer_df = compute_layer_influence(importance_df)
    # print(layer_df)

    # --- Step 7: Anomaly analysis ---
    xct_volume = np.load(XCT_VOLUME_PATH)   # shape: (N_slices, H, W)
    # Build XCT-layer → CSV-filename mapping (adjust offsets for your dataset)
    xct_to_csv = {
        36 + i: f"L{i + 1:04d}.csv"
        for i in range(min(len(dfs), xct_volume.shape[0] - XCT_LAYER_OFFSET))
    }
    final_results = run_anomaly_analysis(
        xct_to_csv, dfs, xct_volume, model, feat_scaler
    )
    final_results.to_csv("anomaly_results.csv", index=False)

    # --- Step 8: Visualize ---
    plot_xct_error_overlay(final_results, xct_volume)
