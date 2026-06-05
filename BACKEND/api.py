#!/usr/bin/env python3
"""
FastAPI Server for Stress Detection
====================================
REST API for real-time stress prediction.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
from typing import List, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Use Transformer model instead of CNN‑LSTM
from models.cnn_transformer import CNNTransformer  # NEW

from utils.signal_filters import (
    filter_ecg,
    filter_eda,
    filter_temp,
    filter_acc,
    resample_signal,
    normalize_zscore,
    compute_acc_magnitude,
)

# ============================================================================
# Configuration
# ============================================================================

# Use new Transformer checkpoint
MODEL_PATH = os.environ.get("MODEL_PATH", "models/cnn_transformer.pth")
DEVICE = os.environ.get("DEVICE", "auto")
TARGET_FS = 32.0   # Target sampling frequency
WINDOW_SAMPLES = 320  # 10 seconds at 32 Hz

# ============================================================================
# Model loading helper
# ============================================================================

def load_model(model_path: str, device: str = "cpu"):
    """
    Load CNNTransformer model from disk.
    """
    # This matches the light config you trained (2 epochs).
    model = CNNTransformer(
        in_channels=4,
        conv_channels=(32, 64),
        conv_kernel_size=5,
        conv_pool=2,
        d_model=64,
        nhead=2,
        num_transformer_layers=1,
        dim_feedforward=128,
        dropout=0.3,
        num_classes=2,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# ============================================================================
# Initialize App
# ============================================================================

app = FastAPI(
    title="Stress Detection API",
    description="Real-time stress detection from physiological signals using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State
# ============================================================================

class AppState:
    model = None
    device = None
    class_names = ["Non-Stress", "Stress"]

state = AppState()

# ============================================================================
# Request/Response Models
# ============================================================================

class SignalData(BaseModel):
    """Raw physiological signal data."""
    ecg: List[float] = Field(..., description="ECG signal samples")
    eda: List[float] = Field(..., description="EDA signal samples")
    temp: List[float] = Field(..., description="Temperature signal samples")
    acc_x: Optional[List[float]] = Field(None, description="Accelerometer X-axis")
    acc_y: Optional[List[float]] = Field(None, description="Accelerometer Y-axis")
    acc_z: Optional[List[float]] = Field(None, description="Accelerometer Z-axis")
    acc_mag: Optional[List[float]] = Field(
        None, description="Accelerometer magnitude (if 3-axis not provided)"
    )

    # Sampling frequencies
    fs_ecg: float = Field(700.0, description="ECG sampling frequency (Hz)")
    fs_eda: float = Field(4.0, description="EDA sampling frequency (Hz)")
    fs_temp: float = Field(4.0, description="Temperature sampling frequency (Hz)")
    fs_acc: float = Field(32.0, description="Accelerometer sampling frequency (Hz)")

    @validator("acc_mag", always=True)
    def validate_acc(cls, v, values):
        has_3axis = all(values.get(k) is not None for k in ["acc_x", "acc_y", "acc_z"])
        if v is None and not has_3axis:
            raise ValueError(
                "Either acc_mag or all of (acc_x, acc_y, acc_z) must be provided"
            )
        return v


class PreprocessedData(BaseModel):
    """Preprocessed signal window data."""
    data: List[List[float]] = Field(
        ...,
        description="Preprocessed data array of shape (channels, time_steps)",
    )

    @validator("data")
    def validate_shape(cls, v):
        if len(v) != 4:
            raise ValueError(f"Expected 4 channels, got {len(v)}")
        return v


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: int = Field(..., description="Predicted class (0=Non-Stress, 1=Stress)")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    n_samples: int
    stress_ratio: float = Field(..., description="Ratio of stress predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    timestamp: str
    version: str

# ============================================================================
# Helper Functions
# ============================================================================

def preprocess_signals(
    ecg: np.ndarray,
    eda: np.ndarray,
    temp: np.ndarray,
    acc: np.ndarray,
    fs_ecg: float,
    fs_eda: float,
    fs_temp: float,
    fs_acc: float,
) -> np.ndarray:
    """Preprocess raw signals."""
    # Filter
    ecg_filt = filter_ecg(ecg, fs_ecg)
    eda_filt = filter_eda(eda, fs_eda)
    temp_filt = filter_temp(temp)
    acc_filt = filter_acc(acc)

    # Resample
    ecg_res = resample_signal(ecg_filt, fs_ecg, TARGET_FS)
    eda_res = resample_signal(eda_filt, fs_eda, TARGET_FS)
    temp_res = resample_signal(temp_filt, fs_temp, TARGET_FS)
    acc_res = resample_signal(acc_filt, fs_acc, TARGET_FS)

    # Truncate to same length
    min_len = min(len(ecg_res), len(eda_res), len(temp_res), len(acc_res))

    # Normalize
    ecg_norm = normalize_zscore(ecg_res[:min_len])
    eda_norm = normalize_zscore(eda_res[:min_len])
    temp_norm = normalize_zscore(temp_res[:min_len])
    acc_norm = normalize_zscore(acc_res[:min_len])

    return np.stack([ecg_norm, eda_norm, temp_norm, acc_norm], axis=0).astype(
        np.float32
    )


def predict_single(x: np.ndarray) -> Dict:
    """Make prediction on single preprocessed sample."""
    if x.ndim == 2:
        x = x[np.newaxis, ...]

    x_tensor = torch.from_numpy(x).float().to(state.device)

    with torch.no_grad():
        logits = state.model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

    pred_idx = pred[0].item()
    prob_values = probs[0].cpu().numpy()

    return {
        "prediction": pred_idx,
        "class_name": state.class_names[pred_idx],
        "confidence": float(prob_values[pred_idx]),
        "probabilities": {
            "Non-Stress": float(prob_values[0]),
            "Stress": float(prob_values[1]),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global state

    # Set device
    if DEVICE == "auto":
        if torch.cuda.is_available():
            state.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            state.device = torch.device("mps")
        else:
            state.device = torch.device("cpu")
    else:
        state.device = torch.device(DEVICE)

    print(f"Using device: {state.device}")

    # Load model
    if os.path.exists(MODEL_PATH):
        try:
            state.model = load_model(MODEL_PATH, str(state.device))
            state.model.eval()
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            state.model = None
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
        state.model = None

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Stress Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        device=str(state.device) if state.device else "unknown",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_preprocessed(data: PreprocessedData):
    """
    Predict stress from preprocessed signal data.
    Input: array of shape (4, time_steps) with channels [ECG, EDA, TEMP, ACC].
    """
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        x = np.array(data.data, dtype=np.float32)
        result = predict_single(x)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )

# ---------- CSV file-upload endpoint for dashboard ----------

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """
    Accept a CSV file from the frontend, convert it to a 4xWINDOW_SAMPLES window,
    run the model, and return label + probability for Stress.
    """
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        # Read uploaded CSV into memory
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Expect at least 4 columns; use the first four as ECG, EDA, TEMP, ACC.
        if df.shape[1] < 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected at least 4 columns, got {df.shape[1]}",
            )

        data_4 = df.iloc[:, :4].to_numpy().T.astype("float32")  # (4, T)

        # Adjust length to WINDOW_SAMPLES
        if data_4.shape[1] >= WINDOW_SAMPLES:
            data_4 = data_4[:, :WINDOW_SAMPLES]
        else:
            pad_len = WINDOW_SAMPLES - data_4.shape[1]
            last_cols = np.repeat(data_4[:, -1:], pad_len, axis=1)
            data_4 = np.concatenate([data_4, last_cols], axis=1)

        # Run model
        result = predict_single(data_4)

        # Map to simple label + probability for dashboard.tsx
        prob_stress = result["probabilities"]["Stress"]
        label = "Stress" if result["prediction"] == 1 else "No Stress (Baseline)"

        return {
            "label": label,
            "probability": float(prob_stress),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File prediction failed: {str(e)}",
        )

# -------------------------------------------------------------------

@app.post("/predict/raw", response_model=BatchPredictionResponse)
async def predict_raw_signals(signals: SignalData):
    """
    Predict stress from raw physiological signals.
    Automatically preprocesses signals and segments into windows.
    """
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        # Convert to numpy
        ecg = np.array(signals.ecg)
        eda = np.array(signals.eda)
        temp = np.array(signals.temp)

        # Handle accelerometer
        if signals.acc_mag is not None:
            acc = np.array(signals.acc_mag)
        else:
            acc = compute_acc_magnitude(
                np.array(signals.acc_x),
                np.array(signals.acc_y),
                np.array(signals.acc_z),
            )

        # Preprocess
        processed = preprocess_signals(
            ecg,
            eda,
            temp,
            acc,
            signals.fs_ecg,
            signals.fs_eda,
            signals.fs_temp,
            signals.fs_acc,
        )

        # Segment into windows
        n_windows = processed.shape[1] // WINDOW_SAMPLES

        if n_windows == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Signal too short. Need at least {WINDOW_SAMPLES} samples after preprocessing",
            )

        predictions: List[PredictionResponse] = []
        for i in range(n_windows):
            start = i * WINDOW_SAMPLES
            end = start + WINDOW_SAMPLES
            window = processed[:, start:end]
            result = predict_single(window)
            predictions.append(PredictionResponse(**result))

        # Compute stress ratio
        stress_count = sum(1 for p in predictions if p.prediction == 1)
        stress_ratio = stress_count / len(predictions)

        return BatchPredictionResponse(
            predictions=predictions,
            n_samples=len(predictions),
            stress_ratio=stress_ratio,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(data: List[PreprocessedData]):
    """
    Batch prediction on multiple preprocessed samples.
    """
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        predictions: List[PredictionResponse] = []
        for sample in data:
            x = np.array(sample.data, dtype=np.float32)
            result = predict_single(x)
            predictions.append(PredictionResponse(**result))

        stress_count = sum(1 for p in predictions if p.prediction == 1)
        stress_ratio = stress_count / len(predictions) if predictions else 0.0

        return BatchPredictionResponse(
            predictions=predictions,
            n_samples=len(predictions),
            stress_ratio=stress_ratio,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
