import io
import os

import numpy as np
import pandas as pd
import torch
import streamlit as st

from cnn_lstm import CNNLSTM


# ---------------------------
# CONFIG
# ---------------------------

MODEL_PATH = "models/cnn_bilstm_weighted.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPECTED_CHANNELS = 4
EXPECTED_TIMESTEPS = 7000

USERS = {
    "admin@example.com": {
        "password": "admin123",
        "role": "admin",
        "name": "Admin",
    },
    "user@example.com": {
        "password": "user123",
        "role": "user",
        "name": "User",
    },
}


# ---------------------------
# MODEL LOADING
# ---------------------------

@st.cache_resource
def load_model():
    model = CNNLSTM(
        in_channels=EXPECTED_CHANNELS,
        num_classes=2,
        bidirectional=True,
    ).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_input_from_csv(uploaded_file: io.BytesIO) -> np.ndarray:
    df = pd.read_csv(uploaded_file)

    if df.shape[1] < EXPECTED_CHANNELS:
        raise ValueError(
            f"CSV must have at least {EXPECTED_CHANNELS} columns "
            f"(found {df.shape[1]})."
        )

    df = df.iloc[:, :EXPECTED_CHANNELS]

    if df.shape[0] < EXPECTED_TIMESTEPS:
        raise ValueError(
            f"CSV must have at least {EXPECTED_TIMESTEPS} rows (time steps); "
            f"found {df.shape[0]}."
        )
    elif df.shape[0] > EXPECTED_TIMESTEPS:
        df = df.iloc[:EXPECTED_TIMESTEPS, :]

    arr = df.to_numpy(dtype=np.float32)  # (time, channels)
    arr = arr.T  # -> (channels, time)
    return arr


def predict_stress(model: CNNLSTM, window_4x7000: np.ndarray):
    x = torch.tensor(
        window_4x7000[None, :, :], dtype=torch.float32, device=DEVICE
    )

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    prob_baseline = float(probs[0])
    prob_stress = float(probs[1])
    return pred_class, prob_baseline, prob_stress


# ---------------------------
# AUTH + PAGES
# ---------------------------

def show_login():
    st.title("🔐 StressWatch AI – Login")

    with st.container():
        st.write("Please sign in to continue.")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = USERS.get(email)
            if user and user["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["user_email"] = email
                st.session_state["user_role"] = user["role"]
                st.session_state["user_name"] = user["name"]
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Invalid email or password.")


def show_user_page(model):
    st.title("💗 StressWatch AI – User")

    st.write(
        "Upload a **4‑channel CSV** (one 10‑second window of ECG, respiration, EMG, etc.) "
        "to see whether the pattern looks more like **Stress** or "
        "**No Stress (Baseline)**."
    )

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("1. Upload your recording 📂")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with 4 columns and 7000 rows",
            type=["csv"],
        )

        if uploaded_file is None:
            st.info("Please upload a CSV file to start.")
            return

        try:
            window = prepare_input_from_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        st.success(
            f"File loaded successfully. Window shape: {window.shape} "
            f"(channels, time steps)"
        )

        with st.expander("Preview first 10 time steps (table)"):
            df_preview = pd.DataFrame(
                window.T[:10, :],
                columns=[f"ch_{i+1}" for i in range(EXPECTED_CHANNELS)],
            )
            st.dataframe(df_preview, use_container_width=True)

        st.subheader("2. Run prediction 🧠")

        if st.button("Predict Stress / No Stress"):
            pred_class, prob_baseline, prob_stress = predict_stress(model, window)

            if pred_class == 1:
                label = "Stress"
                emoji = "⚠️"
            else:
                label = "No Stress (Baseline)"
                emoji = "🌿"

            st.markdown("### Result")
            st.write(f"{emoji} **Predicted label:** {label}")
            st.write(f"🔥 **Probability of Stress:** `{prob_stress:.3f}`")
            st.write(f"💧 **Probability of Baseline:** `{prob_baseline:.3f}`")
            st.progress(min(max(prob_stress, 0.0), 1.0))

            if pred_class == 1:
                st.warning(
                    "This 10‑second window looks more like a **stress pattern**."
                )
                st.markdown(
                    """
**Gentle suggestions (not medical advice):**

- Take 5 slow, deep breaths – in through the nose, out through the mouth.  
- Roll your shoulders and relax your jaw; muscle tension often increases with stress.  
- If intense stress happens often, consider talking with a trusted person or health professional.
"""
                )
            else:
                st.success(
                    "Nice! This window looks more like a **calm / baseline pattern**. 😌"
                )
                st.markdown(
                    """
**Keep it up:**

- Whatever you are doing right now seems to help your body stay calmer.  
- Continue habits like good sleep, small breaks, and gentle movement.  
- Not every moment is calm, so noticing peaceful moments is a big win. 🌟
"""
                )

    with col_right:
        st.subheader("How to read this 🔎")
        st.markdown(
            """
- **Stress**: Higher heart rate / breathing / muscle activity pattern.  
- **No Stress (Baseline)**: Signals look more relaxed in this 10‑second slice.  
- This is a **support tool**, not a medical diagnosis.
"""
        )

        st.subheader("Live signal view 📈")
        st.write("Below is Channel 1 from your uploaded file.")
        if "window" in locals():
            st.line_chart(window[0])
        else:
            st.caption("Upload a file to see the signal.")


def show_admin_page():
    st.title("🛠️ StressWatch AI – Admin")

    st.info(
        "Admin dashboard placeholder. Here you can later show logs, counts, and other stats."
    )

    st.markdown(
        """
Ideas for future admin features:

- Number of uploads per day.  
- Count of Stress vs Baseline predictions.  
- Simple plots of probabilities over time.
"""
    )


def main():
    st.set_page_config(page_title="StressWatch AI", layout="wide")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user_role"] = None
        st.session_state["user_email"] = None
        st.session_state["user_name"] = None

    if not st.session_state["logged_in"]:
        show_login()
        return

    top1, top2, top3 = st.columns([3, 2, 1])
    with top1:
        st.write(f"👤 Logged in as: **{st.session_state['user_name']}**")
    with top2:
        st.write(f"🔑 Role: **{st.session_state['user_role']}**")
    with top3:
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["user_role"] = None
            st.session_state["user_email"] = None
            st.session_state["user_name"] = None
            st.rerun()

    role = st.session_state["user_role"]

    if role == "admin":
        page = st.sidebar.selectbox("Navigation", ["User page", "Admin page"])
    else:
        page = "User page"

    model = load_model()

    if page == "User page":
        show_user_page(model)
    elif page == "Admin page":
        show_admin_page()


if __name__ == "__main__":
    main()
