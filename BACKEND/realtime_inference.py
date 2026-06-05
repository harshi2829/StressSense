import os
import numpy as np
import torch

from cnn_lstm import CNNLSTM   # uses your attention BiLSTM model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SECONDS = 10
CHEST_FS = 700                      # Hz, same as in data_preprocess.py
WINDOW_SAMPLES = WINDOW_SECONDS * CHEST_FS  # 7000
NUM_CHANNELS = 4                    # ECG, EDA, EMG, RESP


class RealTimeStressDetector:
    """
    Maintains a rolling buffer of 4-channel data and runs the
    CNN-BiLSTM-Attention model on each full 10-second window.
    """

    def __init__(self, model_path="models/cnn_bilstm_weighted.pth"):
        # load model
        self.model = CNNLSTM(
            in_channels=NUM_CHANNELS,
            num_classes=2,
            bidirectional=True
        ).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # rolling buffer: shape (4, WINDOW_SAMPLES)
        self.buffer = np.zeros((NUM_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
        self.current_index = 0      # where to write next sample
        self.is_full = False        # becomes True after first 7000 samples

        print(f"RealTimeStressDetector loaded from {model_path} on {DEVICE}.")

    def add_samples(self, new_samples):
        """
        Add new samples to the buffer.

        new_samples: numpy array of shape (NUM_CHANNELS, N_new)
                     channels order must match training (ECG, EDA, EMG, RESP)

        Returns:
            prediction, prob_stress if a full window is ready
            else (None, None)
        """
        if not isinstance(new_samples, np.ndarray):
            new_samples = np.array(new_samples, dtype=np.float32)

        # ensure shape (channels, time)
        if new_samples.ndim == 1:
            # assume same value for all channels (not typical)
            new_samples = np.repeat(new_samples[None, :], NUM_CHANNELS, axis=0)
        elif new_samples.shape[0] != NUM_CHANNELS:
            raise ValueError(
                f"new_samples must have shape ({NUM_CHANNELS}, N), got {new_samples.shape}"
            )

        n_new = new_samples.shape[1]
        for i in range(n_new):
            # add one time step at a time
            self.buffer[:, self.current_index] = new_samples[:, i]
            self.current_index += 1

            if self.current_index >= WINDOW_SAMPLES:
                self.current_index = 0
                self.is_full = True

        # if we do not yet have a full window, no prediction
        if not self.is_full:
            return None, None

        # if buffer is "full", run prediction on the current buffer
        return self._predict_current_window()

    def _predict_current_window(self):
        """
        Run model on the current buffer (4, 7000) and return prediction.
        """
        # buffer is (channels, time), model expects (batch, channels, time)
        x = torch.tensor(self.buffer[None, :, :], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = self.model(x)          # (1, 2)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (2,)

        # class 0 = baseline, 1 = stress
        pred_class = int(np.argmax(probs))
        prob_stress = float(probs[1])

        return pred_class, prob_stress


if __name__ == "__main__":
    # Example: simulate streaming from a recorded window
    # Here we just load one window from your dataset and feed it in pieces.

    # make sure you have data_processed/X.npy
    X = np.load(os.path.join("data_processed", "X.npy"))  # (N, 4, 7000)
    print("Loaded X.npy with shape:", X.shape)

    detector = RealTimeStressDetector("models/cnn_bilstm_weighted.pth")

    # take the first window and pretend it arrives 1 sample at a time
    window = X[0]  # shape (4, 7000)

    # send in chunks of, e.g., 50 samples
    chunk_size = 50
    for start in range(0, window.shape[1], chunk_size):
        end = min(start + chunk_size, window.shape[1])
        chunk = window[:, start:end]     # (4, chunk_size)

        pred, prob_stress = detector.add_samples(chunk)
        if pred is not None:
            label = "Stress" if pred == 1 else "Baseline"
            print(f"Real-time prediction: {label}, prob_stress={prob_stress:.3f}")
