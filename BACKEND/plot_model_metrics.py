import numpy as np
import matplotlib.pyplot as plt

models = ["CNN", "LSTM", "Attention", "Transformer"]

accuracy  = [0.91, 0.93, 0.94, 0.95]
precision = [0.90, 0.92, 0.93, 0.94]
f1_score  = [0.89, 0.91, 0.93, 0.94]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))

rects1 = ax.bar(x - width, accuracy,  width, label="Accuracy")
rects2 = ax.bar(x,         precision, width, label="Precision")
rects3 = ax.bar(x + width, f1_score,  width, label="F1-score")

ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("Model Performance on Stress Detection")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

for rects in (rects1, rects2, rects3):
    for r in rects:
        h = r.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.show()
