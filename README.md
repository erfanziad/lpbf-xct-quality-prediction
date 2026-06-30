# LPBF XCT Quality Prediction

A deep learning pipeline for **voxel-level XCT quality prediction** in Laser Powder Bed Fusion (LPBF) additive manufacturing, using multimodal in-situ monitoring data and a spatio-temporal Transformer architecture.

---

## Overview

This project fuses three complementary in-situ data streams — melt pool camera imagery, layerwise optical imaging (LED flash), and process parameter logs — to predict post-build XCT pixel grayscale values at the voxel level. Spatial neighbors and multi-layer temporal context are explicitly encoded in the feature representation, enabling the model to capture how local printing history influences subsurface material quality.

The final trained model supports calibrated anomaly detection: each voxel is classified as well-predicted, under-predicted, or over-predicted relative to its XCT ground truth, and results are overlaid directly on XCT slices for interpretability.

---

## Pipeline

```
CSV layers (L0001 – L0250)
        │
        ├─ [Step 1] Load per-layer process + sensor data
        │
        ├─ [Step 2] Extract 12 melt-pool image features per frame
        │            (area, entropy, GLCM energy, convex hull, etc.)
        │
        ├─ [Step 3] DNN-based melt-pool event classification
        │            → Predicted_Category, Spatter, Tail, Plume
        │
        ├─ [Step 4] Build neighbor-augmented feature matrix
        │            · Spatial neighbors (ε-radius) in current layer
        │            · 5 layers above  +  2 layers below
        │
        ├─ [Step 5] Train Transformer Regressor
        │            → predict XCT pixel grayscale (5×5×5)
        │
        ├─ [Step 6] Permutation feature importance + layer-influence analysis
        │
        └─ [Step 7] Calibrated anomaly detection + XCT overlay visualization
```

---

## Repository Structure

```
lpbf-xct-quality-prediction/
│
├── lpbf_xct_prediction.py   # Main pipeline (all steps in one script)
├── README.md
├── requirements.txt
│
└── outputs/                 # Generated at runtime
    ├── multilayer_features.csv
    ├── transformer_model.pth
    └── anomaly_results.csv
```

---

## Requirements

```
python >= 3.10
numpy
pandas
scikit-learn
scipy
torch >= 2.0
opencv-python
scikit-image
matplotlib
seaborn
shap
tqdm
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

This pipeline is designed for the **Zhou et al. LPBF registered dataset**, which contains per-layer CSV files with the following channels:

- Process parameters: laser power, scan speed, X/Y position (commanded and real)
- Melt pool geometry: length, width, area at three intensity thresholds (80 / 100 / 120)
- Layerwise pixel grayscale: LED a/b/c, single pixel and 3×3 / 5×5 neighborhoods, with and without burned-region correction
- XCT pixel grayscale: single voxel and 3×3×3 / 5×5×5 neighborhoods (ground truth)

Melt pool camera images are expected at:
```
<IMAGE_BASE_DIR>/Stack_NNN/frame_MMMMM.png
```

The 3-D XCT volume should be a NumPy array of shape `(N_slices, H, W)` saved as `.npy`.

---

## Usage

### 1. Configure paths

Edit the configuration block at the top of `lpbf_xct_prediction.py`:

```python
CSV_DIR        = "/path/to/csv_layers"
IMAGE_BASE_DIR = "/path/to/tiff_stacks"
XCT_VOLUME_PATH = "/path/to/xct_volume.npy"
```

### 2. Run the full pipeline

```bash
python lpbf_xct_prediction.py
```

The preprocessing steps (image feature extraction and DNN inference) are commented out by default in `main` since they modify CSVs in-place and only need to run once. Uncomment them on first run:

```python
# Step 2 — run once
append_image_features_to_csvs(CSV_DIR, IMAGE_BASE_DIR)

# Step 3 — run once (requires a pretrained MeltPoolDNN checkpoint)
run_dnn_inference(CSV_DIR, model_path="feature_based_dnn.pth")
```

### 3. Outputs

| File | Description |
|---|---|
| `multilayer_features.csv` | Neighbor-augmented feature matrix |
| `transformer_model.pth` | Trained Transformer checkpoint + scalers |
| `anomaly_results.csv` | Per-voxel predictions with calibrated error labels |

---

## Model Architecture

**TransformerRegressor** — flat-feature variant:

```
Input (D_in)
    → Linear projection → hidden_dim (128)
    → Transformer Encoder (2 layers, 8 heads, GELU)
    → Regression head (128 → 64 → 1)
```

Training uses a 70 / 15 / 15 train/val/test split, MSE loss, Adam optimizer, and early stopping on validation loss. The target (XCT grayscale) is IQR-cleaned and Min-Max scaled before training.

**MeltPoolDNN** — 4-output binary classifier:

```
Input (12 image features)
    → Linear(1024) → BN → ReLU → Dropout(0.4)
    → Linear(512)  → BN → ReLU → Dropout(0.4)
    → Linear(256)  → BN → ReLU → Dropout(0.4)
    → Linear(4)    → Sigmoid
```
Outputs: `Predicted_Category`, `Predicted_Spatter`, `Predicted_Tail`, `Predicted_Plume`

---

## Feature Importance

Permutation importance is computed over the test set and aggregated two ways:

- **Per-feature** — which physical signal (melt pool geometry, layerwise grayscale, DNN label, etc.) contributes most
- **Per-layer** — how much temporal depth (current, 1–5 above, 1–2 below) influences prediction accuracy

---

## Anomaly Detection

After training, a bias calibration step estimates the mean systematic offset between predictions and ground truth. A 3σ threshold is then applied to the calibrated error distribution to label each voxel:

| Label | Condition |
|---|---|
| `well_predicted` | \|calibrated error\| ≤ 3σ |
| `under_predicted` | calibrated error > +3σ |
| `over_predicted` | calibrated error < −3σ |

Results are overlaid on the corresponding XCT slices with color-coded scatter points.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ziaeidad2025lpbf,
  author       = {Erfan Ziaei-Rad},
  title        = {LPBF XCT Quality Prediction via Spatio-Temporal Transformer},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/erfanziad/lpbf-xct-quality-prediction}
}
```

---

## License

MIT License. See `LICENSE` for details.
