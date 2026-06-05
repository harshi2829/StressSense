from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import numpy as np
import torch

from realtime_inference import RealTimeStressDetector

# ---------------------------
# INIT APP AND MODEL
# ---------------------------

app = FastAPI(title="Stress Detection API")

# load your real-time detector (uses cnn_bilstm_weighted.pth)
detector = RealTimeStressDetector("models/cnn_bilstm_weighted.pth")


# ---------------------------
# REQUEST/RESPONSE MODELS
# ---------------------------

class WindowRequest(BaseModel):
    # single 10-second window: 4 channels x 7000 samples
    # data is flattened 1D list: length = 4 * 7000
    window: List[float]


class WindowResponse(BaseModel):
    label: str          # "stress" or "baseline"
    prob_stress: float  # probability for stress (0-1)


class StreamChunkRequest(BaseModel):
    # small chunk of data, e.g. 4 x N samples (flattened)
    # example: N = 50 -> list length = 200
    chunk: List[float]
    n_samples: int      # number of time steps (N)


class StreamChunkResponse(BaseModel):
    label: str | None       # "stress" or "baseline", or None if not enough data yet
    prob_stress: float | None


# ---------------------------
# ENDPOINTS
# ---------------------------

@app.get("/")
def root():
    return {"message": "Stress Detection API is running."}


@app.post("/predict_window", response_model=WindowResponse)
def predict_window(req: WindowRequest):
    """
    Predict from one full window (4 x 7000 samples).
    Use when you already have a full 10-second chunk.
    """
    arr = np.array(req.window, dtype=np.float32)

    # reshape to (4, 7000)
    expected_len = 4 * 7000
    if arr.size != expected_len:
        return {
            "label": "error",
            "prob_stress": 0.0,
        }

    window = arr.reshape(4, 7000)

    # run model: single window
    x = torch.tensor(window[None, :, :], dtype=torch.float32).to(detector.model.device)
    with torch.no_grad():
        logits = detector.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))    # 0 baseline, 1 stress
    prob_stress = float(probs[1])

    label = "stress" if pred_class == 1 else "baseline"
    return WindowResponse(label=label, prob_stress=prob_stress)


@app.post("/predict_stream", response_model=StreamChunkResponse)
def predict_stream(req: StreamChunkRequest):
    """
    Real-time style: send small chunks repeatedly.
    The API maintains a rolling 10-second buffer internally.
    When the buffer is full, returns a prediction; otherwise returns nulls.
    """
    arr = np.array(req.chunk, dtype=np.float32)

    # reshape to (4, N)
    if arr.size != 4 * req.n_samples:
        return StreamChunkResponse(label=None, prob_stress=None)

    chunk = arr.reshape(4, req.n_samples)

    pred, prob_stress = detector.add_samples(chunk)
    if pred is None:
        # not enough data yet
        return StreamChunkResponse(label=None, prob_stress=None)

    label = "stress" if pred == 1 else "baseline"
    return StreamChunkResponse(label=label, prob_stress=prob_stress)
