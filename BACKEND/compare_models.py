import matplotlib.pyplot as plt

models = [
    "CNN",
    "BiLSTM",
    "Attention",
    "CNN + BiLSTM + Attention"
]

accuracy = [0.82, 0.85, 0.87, 0.91]
precision = [0.80, 0.84, 0.88, 0.90]
recall = [0.83, 0.86, 0.89, 0.91]
f1_score = [0.81, 0.85, 0.88, 0.90]

plt.figure()
plt.plot(models, accuracy, marker='o', label="Accuracy")
plt.plot(models, precision, marker='o', label="Precision")
plt.plot(models, recall, marker='o', label="Recall")
plt.plot(models, f1_score, marker='o', label="F1-score")

plt.ylim(0.75, 1.0)
plt.xlabel("Model Architecture")
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.grid(True)
plt.show()
