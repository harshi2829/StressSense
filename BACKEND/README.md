# Stress Detection System

A deep learning backend for binary stress detection using physiological signals from the WESAD dataset.

## Overview

This system classifies stress levels into two categories using physiological signals:
- **Non-Stress** (baseline, amusement, meditation)
- **Stress** (stress condition)

### Features

- 🧠 1D-CNN model (StressCNN) for binary classification
- 📊 Complete preprocessing pipeline for physiological signals
- 🚀 FastAPI inference server with health check and prediction endpoints
- 📁 Modular, production-ready code structure

## Project Structure

```
project/
├── configs/
│   ├── preprocess_config.yaml    # Preprocessing parameters
│   └── training_config.yaml      # Training hyperparameters
├── data_raw/                     # Raw WESAD data (you provide)
│   ├── S2/
│   │   └── S2.pkl
│   ├── S3/
│   │   └── S3.pkl
│   └── ...
├── data_processed/               # Preprocessed tensors (generated)
│   ├── X.npy                     # Signal windows
│   └── y.npy                     # Labels
├── models/
│   ├── __init__.py
│   ├── cnn_model.py              # StressCNN architecture
│   └── stress_cnn.pth            # Trained model (generated)
├── utils/
│   ├── __init__.py
│   ├── preprocess.py             # Preprocessing functions
│   ├── data_loader.py            # PyTorch Dataset/DataLoader
│   └── signal_filters.py         # Signal filtering functions
├── data_preprocess.py            # CLI preprocessing script
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── api.py                        # FastAPI server
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download WESAD Dataset

Download the WESAD dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) and extract to `data_raw/`.

Expected structure:
```
data_raw/
├── S2/
│   └── S2.pkl
├── S3/
│   └── S3.pkl
├── S4/
│   └── S4.pkl
...
```

## Usage

### Step 1: Preprocess Data

Run the preprocessing pipeline to convert raw signals to windowed tensors:

```bash
# Basic usage
python data_preprocess.py --input data_raw/ --output data_processed/

# With custom parameters
python data_preprocess.py \
    --input data_raw/ \
    --output data_processed/ \
    --window-size 10 \
    --sampling-rate 32 \
    --normalize zscore

# Process specific subjects
python data_preprocess.py \
    --input data_raw/ \
    --output data_processed/ \
    --subjects S2 S3 S4

# Using config file
python data_preprocess.py --config configs/preprocess_config.yaml
```

**Output:**
- `data_processed/X.npy` - Shape: (n_windows, 4, 320) = 4 channels × 10s × 32Hz
- `data_processed/y.npy` - Shape: (n_windows,) - Binary labels (0 or 1)

### Step 2: Train Model

Train the 1D-CNN model on preprocessed data:

```bash
# Basic training
python train.py

# With custom parameters
python train.py \
    --data data_processed/ \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001

# Using config file
python train.py --config configs/training_config.yaml
```

**Output:**
- `models/stress_cnn.pth` - Trained model weights

### Step 3: Run Predictions

Use the trained model for inference:

```bash
# Predict on test data
python predict.py --model models/stress_cnn.pth --data data_processed/X.npy

# Predict on specific sample (by index)
python predict.py --model models/stress_cnn.pth --data data_processed/X.npy --sample 0
```

### Step 4: Start API Server

Run the FastAPI inference server:

```bash
# Start server
python api.py

# Or using uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Predict stress level |

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Prediction Request:**
```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "signals": [[...4 channels × 320 samples...]],
        "sampling_rate": 32
    }'
```

**Response:**
```json
{
    "prediction": 1,
    "label": "Stress",
    "probability": 0.85,
    "probabilities": {
        "Non-Stress": 0.15,
        "Stress": 0.85
    }
}
```

## Preprocessing Pipeline

The preprocessing pipeline includes:

1. **Signal Loading**: Load raw pickle files from WESAD dataset
2. **Signal Extraction**: Extract ECG, EDA, TEMP, ACC from chest sensors
3. **Filtering**:
   - ECG: Bandpass filter (0.5-40 Hz)
   - EDA: Lowpass filter (5 Hz)
   - TEMP: Moving average smoothing
   - ACC: Magnitude calculation + smoothing
4. **Resampling**: Resample all signals to 32 Hz
5. **Label Mapping**: Map WESAD labels to binary (stress vs non-stress)
6. **Windowing**: Create non-overlapping 10-second windows
7. **Normalization**: Z-score normalization per window per channel

## Model Architecture

StressCNN - 1D-CNN for binary classification:

```
Input: (batch, 4 channels, 320 samples)
  ↓
Conv1D(4→64, k=7) + BatchNorm + ReLU + MaxPool(2)
  ↓
Conv1D(64→128, k=5) + BatchNorm + ReLU + MaxPool(2)
  ↓
Conv1D(128→256, k=3) + BatchNorm + ReLU + MaxPool(2)
  ↓
Global Average Pooling
  ↓
Dropout(0.5)
  ↓
Linear(256→128) + ReLU
  ↓
Linear(128→2)
  ↓
Output: 2 classes (Non-Stress, Stress)
```

## Configuration Files

### preprocess_config.yaml

```yaml
input: "data_raw/"
output: "data_processed/"
window_size: 10.0
overlap: 0.0
sampling_rate: 32
normalize: "zscore"

filtering:
  ecg:
    lowcut: 0.5
    highcut: 40.0
    order: 4
  eda:
    cutoff: 5.0
    order: 4
  temp:
    window_size: 5
  acc:
    window_size: 5
```

### training_config.yaml

```yaml
data_dir: "data_processed/"
model_save_path: "models/stress_cnn.pth"
epochs: 100
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
early_stopping_patience: 10
validation_split: 0.2
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- FastAPI
- Uvicorn
- PyYAML
- scikit-learn

## License

This project is for educational and research purposes.

## References

- Schmidt, P., et al. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. ICMI 2018.
- [WESAD Dataset](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
