import os
import numpy as np
import subprocess

VENV_PYTHON = os.path.join("venv311", "Scripts", "python.exe")

# -------------------------------------------------
# STATUS
# -------------------------------------------------
def show_status():
    print("\n--- Project Status ---\n")

    files = {
        "Raw data folder": "data_raw",
        "Processed data (X)": "data_processed/X.npy",
        "Processed labels (y)": "data_processed/y.npy",
        "Trained model": "models/stress_cnn_bilstm_attention.h5",
        "Training script": "src/train_model.py",
        "Inference script": "src/infer_model.py",
        "Evaluation script": "src/evaluate_model.py"
    }

    for name, path in files.items():
        print(f"[{'✅' if os.path.exists(path) else '❌'}] {name}")

    try:
        X = np.load("data_processed/X.npy")
        y = np.load("data_processed/y.npy")
        print(f"\nData Loaded Successfully")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    except Exception as e:
        print("\n[⚠️] Could not load processed data")

# -------------------------------------------------
# TRAIN
# -------------------------------------------------
def train_model():
    print("\n[INFO] Training model...\n")
    subprocess.run([VENV_PYTHON, "src/train_model.py"])

# -------------------------------------------------
# INFERENCE
# -------------------------------------------------
def run_inference():
    print("\n[INFO] Running inference...\n")
    subprocess.run([VENV_PYTHON, "src/infer_model.py"])

# -------------------------------------------------
# METRICS
# -------------------------------------------------
def show_metrics():
    print("\n[INFO] Evaluating model...\n")
    subprocess.run([VENV_PYTHON, "src/evaluate_model.py"])

# -------------------------------------------------
# MAIN CONSOLE
# -------------------------------------------------
def main():
    print("\n=== Stress Detection Project Console ===")
    print("Commands:")
    print("  status   -> Show project status")
    print("  train    -> Train the model")
    print("  infer    -> Run inference")
    print("  metrics  -> Accuracy / Precision / F1-score")
    print("  exit     -> Quit\n")

    while True:
        try:
            cmd = input(">>> ").strip().lower()
            if cmd == "status":
                show_status()
            elif cmd == "train":
                train_model()
            elif cmd == "infer":
                run_inference()
            elif cmd == "metrics":
                show_metrics()
            elif cmd == "exit":
                print("Exiting console...")
                break
            else:
                print("Invalid command")
        except KeyboardInterrupt:
            print("\nExiting console...")
            break
 
if __name__ == "__main__":
    main()
