import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data (or use all data if no separate test set)
X = np.load("data_processed/X.npy")
y = np.load("data_processed/y.npy")

# List of models to compare
models = {
    "CNN": "cnn_model.h5",
    "BiLSTM": "bilstm_model.h5",
    "Attention": "attention_model.h5",
    "Combined": "stress_cnn_bilstm_attention.h5"
}

metrics_dict = {
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-score": []
}

# Evaluate each model
for name, path in models.items():
    try:
        print(f"[INFO] Evaluating {name} model...")
        model = load_model(path, compile=False)
        y_pred_prob = model.predict(X)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        metrics_dict["Accuracy"].append(acc)
        metrics_dict["Precision"].append(prec)
        metrics_dict["Recall"].append(rec)
        metrics_dict["F1-score"].append(f1)

        print(f"{name} -> Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}\n")
    except Exception as e:
        print(f"[ERROR] Could not evaluate {name} model: {e}")
        metrics_dict["Accuracy"].append(0)
        metrics_dict["Precision"].append(0)
        metrics_dict["Recall"].append(0)
        metrics_dict["F1-score"].append(0)

# Plot comparison
labels = list(models.keys())
x = np.arange(len(labels))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - 1.5*width, metrics_dict["Accuracy"], width, label="Accuracy")
plt.bar(x - 0.5*width, metrics_dict["Precision"], width, label="Precision")
plt.bar(x + 0.5*width, metrics_dict["Recall"], width, label="Recall")
plt.bar(x + 1.5*width, metrics_dict["F1-score"], width, label="F1-score")

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
