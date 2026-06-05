import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load data
X = np.load("data_processed/X.npy")
y = np.load("data_processed/y.npy")

# List of your trained model files
models_info = {
    'CNN': 'cnn_model.h5',
    'BiLSTM': 'bilstm_model.h5',
    'Attention': 'attention_model.h5',
    'Combined': 'combined_model.h5'
}

metrics = {}

print("\n=== EVALUATING MODELS ===")
for name, path in models_info.items():
    try:
        model = load_model(path)
        y_pred = (model.predict(X) > 0.5).astype(int)
        metrics[name] = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        print(f"{name} evaluated successfully.")
    except Exception as e:
        print(f"Failed to evaluate {name}: {e}")

# Print metrics in terminal
print("\n=== MODEL METRICS ===")
for model, m in metrics.items():
    print(f"{model}: Accuracy={m['accuracy']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")

# Ask if user wants to plot graph
plot_choice = input("\nDo you want to plot the graph? (y/n): ").strip().lower()
if plot_choice == 'y' and metrics:
    models = list(metrics.keys())
    accuracy = [metrics[m]['accuracy'] for m in models]
    precision = [metrics[m]['precision'] for m in models]
    recall = [metrics[m]['recall'] for m in models]
    f1 = [metrics[m]['f1'] for m in models]

    x = range(len(models))
    width = 0.2

    plt.figure(figsize=(10,6))
    plt.bar([i - width*1.5 for i in x], accuracy, width=width, label='Accuracy')
    plt.bar([i - width/2 for i in x], precision, width=width, label='Precision')
    plt.bar([i + width/2 for i in x], recall, width=width, label='Recall')
    plt.bar([i + width*1.5 for i in x], f1, width=width, label='F1-score')

    plt.xticks(x, models)
    plt.ylabel("Scores")
    plt.title("Comparison of CNN, BiLSTM, Attention, and Combined Models")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Graph plotting skipped.")
