import numpy as np

X = np.load("data_processed/X.npy")
y = np.load("data_processed/y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique.tolist(), counts.tolist())))
