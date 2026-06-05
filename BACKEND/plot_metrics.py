import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse

# Load data
X = np.load("data_processed/X.npy")
y = np.load("data_processed/y.npy")

# Flatten y if needed
if len(y.shape) > 1 and y.shape[1] == 1:
    y = y.ravel()

# Model files
model_files = {
    "CNN": "cnn_model.h5",
    "BiLSTM": "bilstm_model.h5",
    "Attention": "attention_model.h5",
    "Combined": "stress_cnn_bilstm_attention.h5"
}

def evaluate_model(model_file):
    model = load_model(model_file)
    preds = model.predict(X)
    preds_binary = (preds > 0.5).astype(int).ravel()

    acc = accuracy_score(y, preds_binary)
    prec = precision_score(y, preds_binary, zero_division=0)
    rec = recall_score(y, preds_binary, zero_division=0)
    f1 = f1_score(y, preds_binary, zero_division=0)
    cm = confusion_matrix(y, preds_binary)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}

def plot_metrics(metrics_dict, models_to_plot):
    x = range(len(models_to_plot))
    width = 0.2

    accuracy = [metrics_dict[m]['accuracy'] for m in models_to_plot]
    precision = [metrics_dict[m]['precision'] for m in models_to_plot]
    recall = [metrics_dict[m]['recall'] for m in models_to_plot]
    f1 = [metrics_dict[m]['f1'] for m in models_to_plot]

    plt.bar([i-width*1.5 for i in x], accuracy, width, label='Accuracy')
    plt.bar([i-width*0.5 for i in x], precision, width, label='Precision')
    plt.bar([i+width*0.5 for i in x], recall, width, label='Recall')
    plt.bar([i+width*1.5 for i in x], f1, width, label='F1-score')

    plt.xticks(x, models_to_plot)
    plt.ylabel("Scores")
    plt.title("Model Metrics Comparison")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Evaluate and plot all models (CNN, BiLSTM, Attention, Combined)")
    parser.add_argument("--combined", action="store_true", help="Evaluate and plot combined model only")
    args = parser.parse_args()

    metrics = {}

    if args.all:
        for name in ["CNN", "BiLSTM", "Attention", "Combined"]:
            print(f"Evaluating {name}...")
            metrics[name] = evaluate_model(model_files[name])
            print(f"{name} Metrics: {metrics[name]}\n")
        plot_metrics(metrics, ["CNN", "BiLSTM", "Attention", "Combined"])

    if args.combined:
        print("Evaluating Combined model...")
        metrics["Combined"] = evaluate_model(model_files["Combined"])
        print(f"Combined Metrics: {metrics['Combined']}\n")
        plot_metrics(metrics, ["Combined"])
