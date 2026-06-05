
import io
from typing import Literal

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from cnn_lstm import CNNLSTM


# ---------------------------
# CONFIG
# ---------------------------

MODEL_PATH = "models/cnn_bilstm_weighted.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPECTED_CHANNELS = 4
EXPECTED_TIMESTEPS = 7000


# ---------------------------
# FASTAPI APP + CORS
# ---------------------------

app = FastAPI(title="StressWatch AI Backend")

# Allow local frontend dev (adjust origins if needed)
origins = [
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# MODEL LOADING
# ---------------------------

def load_model() -> CNNLSTM:
    model = CNNLSTM(
        in_channels=EXPECTED_CHANNELS,
        num_classes=2,
        bidirectional=True,
    ).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()


def prepare_input_from_csv_bytes(contents: bytes) -> np.ndarray:
    """
    Read CSV bytes and return array of shape (4, 7000).
    """
    buffer = io.BytesIO(contents)
    df = pd.read_csv(buffer)

    if df.shape[1] < EXPECTED_CHANNELS:
        raise ValueError(
            f"CSV must have at least {EXPECTED_CHANNELS} columns; found {df.shape[1]}."
        )

    df = df.iloc[:, :EXPECTED_CHANNELS]

    if df.shape[0] < EXPECTED_TIMESTEPS:
        raise ValueError(
            f"CSV must have at least {EXPECTED_TIMESTEPS} rows; found {df.shape[0]}."
        )
    elif df.shape[0] > EXPECTED_TIMESTEPS:
        df = df.iloc[:EXPECTED_TIMESTEPS, :]

    arr = df.to_numpy(dtype=np.float32)  # (time, channels)
    arr = arr.T  # -> (channels, time)
    return arr


def run_model(window_4x7000: np.ndarray):
    x = torch.tensor(
        window_4x7000[None, :, :], dtype=torch.float32, device=DEVICE
    )  # (1, 4, 7000)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    prob_baseline = float(probs[0])
    prob_stress = float(probs[1])
    return pred_class, prob_baseline, prob_stress



@app.get("/")
def root():
    return {"message": "StressWatch AI backend is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a CSV file upload, runs the stress model, and returns JSON.
    Expected CSV: 4 columns, 7000 rows.
    Response JSON format matches what the Lovable frontend needs.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    contents = await file.read()

    try:
        window = prepare_input_from_csv_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    pred_class, prob_baseline, prob_stress = run_model(window)

    label: Literal["stress", "no_stress"] = "stress" if pred_class == 1 else "no_stress"

    return {
        "filename": file.filename,
        "label": label,
        "prob_stress": prob_stress,
        "prob_baseline": prob_baseline,
    }
