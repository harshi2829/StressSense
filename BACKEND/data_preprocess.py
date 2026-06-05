import os
import numpy as np
import pickle


# ---------------------------
# CONFIG
# ---------------------------

DATA_RAW_DIR = "data_raw"
DATA_PROCESSED_DIR = "data_processed"

# subjects to use (you can change this list)
SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8",
            "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

# WESAD chest sampling rate
CHEST_FS = 700  # Hz

WINDOW_SECONDS = 10
WINDOW_SAMPLES = WINDOW_SECONDS * CHEST_FS  # 7000 samples

# Chest layout we build below: ACC_x, ACC_y, ACC_z, ECG, EDA, EMG, RESP, TEMP  -> 8 channels (0..7)
# Your keys are 'ACC', 'ECG', 'EMG', 'EDA', 'Resp', 'Temp'
# We'll map:
#   RESP <- chest['Resp']
#   TEMP <- chest['Temp']
# We will use ECG, EDA, EMG, RESP as 4 channels: indices 3,4,5,6
CHEST_CHANNELS = [3, 4, 5, 6]

# label mapping: WESAD uses:
# 1 = baseline, 2 = stress, 3 = amusement, etc.
LABEL_BASELINE = 1
LABEL_STRESS = 2


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def load_subject_pkl(subject_id):
    """
    Load WESAD subject .pkl file.
    """
    pkl_path = os.path.join(DATA_RAW_DIR, subject_id, f"{subject_id}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Could not find {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data  # dict with keys like 'signal', 'label', ...


def extract_chest_signals_and_labels(data):
    """
    From loaded subject data, extract:
    - chest signals: channels x time
    - labels: per-sample labels (same length as time)
    """
    chest = data["signal"]["chest"]  # dict
    labels = data["label"]           # 1D array, length = time

    # show keys so you can see them once
    print("Chest keys:", chest.keys())

    # your keys: 'ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp'
    acc = chest["ACC"]          # (time, 3)
    ecg = chest["ECG"].reshape(-1, 1)
    eda = chest["EDA"].reshape(-1, 1)
    emg = chest["EMG"].reshape(-1, 1)
    resp = chest["Resp"].reshape(-1, 1)
    temp = chest["Temp"].reshape(-1, 1)

    # ACC_x, ACC_y, ACC_z, ECG, EDA, EMG, RESP, TEMP -> 8 channels
    chest_all = np.concatenate(
        [acc, ecg, eda, emg, resp, temp], axis=1
    )  # (time, 8)

    chest_all = chest_all.T  # (8, time)

    return chest_all, labels


def make_windows(chest_all, labels):
    """
    Create fixed-length windows from chest signals and labels.
    Returns:
    - X_windows: (N, len(CHEST_CHANNELS), WINDOW_SAMPLES)
    - y_windows: (N,)
    Only keep windows that are dominantly baseline or stress.
    """
    num_samples = chest_all.shape[1]
    step = WINDOW_SAMPLES  # non-overlapping; change to smaller for overlapping if you want

    X_list = []
    y_list = []

    for start in range(0, num_samples - WINDOW_SAMPLES + 1, step):
        end = start + WINDOW_SAMPLES

        window_signals = chest_all[:, start:end]   # (channels_all, WINDOW_SAMPLES)
        window_labels = labels[start:end]          # (WINDOW_SAMPLES,)

        # find the dominant label in this window
        unique, counts = np.unique(window_labels, return_counts=True)
        label_counts = dict(zip(unique, counts))

        # skip if neither baseline nor stress present
        if LABEL_BASELINE not in label_counts and LABEL_STRESS not in label_counts:
            continue

        baseline_count = label_counts.get(LABEL_BASELINE, 0)
        stress_count = label_counts.get(LABEL_STRESS, 0)

        if baseline_count == 0 and stress_count == 0:
            continue

        if stress_count >= baseline_count:
            window_label = 1  # 1 = stress
        else:
            window_label = 0  # 0 = baseline

        # select chosen channels from chest_all
        window_sel = window_signals[CHEST_CHANNELS, :]  # (len(CHEST_CHANNELS), WINDOW_SAMPLES)

        X_list.append(window_sel)
        y_list.append(window_label)

    if len(X_list) == 0:
        return (
            np.empty((0, len(CHEST_CHANNELS), WINDOW_SAMPLES), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X_windows = np.stack(X_list).astype(np.float32)  # (N, len(CHEST_CHANNELS), WINDOW_SAMPLES)
    y_windows = np.array(y_list, dtype=np.int64)     # (N,)

    return X_windows, y_windows


def build_dataset():
    """
    Loop over all subjects, build windows, and save X.npy, y.npy.
    """
    all_X = []
    all_y = []

    for sid in SUBJECTS:
        print(f"Processing {sid}...")
        data = load_subject_pkl(sid)
        chest_all, labels = extract_chest_signals_and_labels(data)
        X_win, y_win = make_windows(chest_all, labels)

        print(f"  {sid}: got {X_win.shape[0]} windows")

        if X_win.shape[0] > 0:
            all_X.append(X_win)
            all_y.append(y_win)

    if len(all_X) == 0:
        print("No windows created. Check your configuration.")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(DATA_PROCESSED_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y.npy"), y)

    print("Saved:")
    print("  ", os.path.join(DATA_PROCESSED_DIR, "X.npy"), X.shape)
    print("  ", os.path.join(DATA_PROCESSED_DIR, "y.npy"), y.shape)


if __name__ == "__main__":
    build_dataset()
