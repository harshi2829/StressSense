import os
import numpy as np
import torch
import streamlit as st

from cnn_lstm import CNNLSTM   # attention BiLSTM model


# ---------------------------
# CONFIG
# ---------------------------

DATA_PROCESSED_DIR = "data_processed"
MODEL_PATH = "models/cnn_bilstm_weighted.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
    model = CNNLSTM(
        in_channels=4,
        num_classes=2,
        bidirectional=True
    ).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource
def load_data():
    X = np.load(os.path.join(DATA_PROCESSED_DIR, "X.npy"))  # (N, 4, 7000)
    y = np.load(os.path.join(DATA_PROCESSED_DIR, "y.npy"))  # (N,)
    return X, y


def predict_window(model, window_4x7000):
    x = torch.tensor(window_4x7000[None, :, :], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))   # 0 baseline, 1 stress
    prob_stress = float(probs[1])
    return pred_class, prob_stress


# ---------------------------
# STREAMLIT UI
# ---------------------------

def main():
    st.title("Real-time Stress Detection (WESAD)")

    st.write(
        "This app uses a CNN–BiLSTM–Attention model trained on the WESAD dataset "
        "to classify 10‑second windows of physiological signals as **baseline** or **stress**."
    )

    model = load_model()
    X, y = load_data()
    N = X.shape[0]

    st.sidebar.header("Window selection")

    idx = st.sidebar.slider("Choose a sample index", min_value=0, max_value=N - 1, value=0)
    if st.sidebar.button("Random sample"):
        idx = np.random.randint(0, N)

    st.write(f"Selected sample index: `{idx}`")
    true_label = "stress" if int(y[idx]) == 1 else "baseline"
    st.write(f"True label from WESAD: **{true_label}**")

    window = X[idx]  # (4, 7000)

    if st.button("Predict stress state"):
        pred, prob_stress = predict_window(model, window)
        pred_label = "stress" if pred == 1 else "baseline"

        st.subheader("Prediction")
        st.write(f"Predicted label: **{pred_label}**")
        st.write(f"Probability of stress: **{prob_stress:.3f}**")

        st.progress(min(max(prob_stress, 0.0), 1.0))

    with st.expander("Show raw signal info"):
        st.write("Window shape:", window.shape)
        st.line_chart(window[0], height=150, width=600)
        st.caption("Channel 1 example (e.g., ECG) for this 10‑second window.")


if __name__ == "__main__":
    main()
