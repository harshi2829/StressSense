import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# NEW: import CNNTransformer from models/cnn_transformer.py
from models.cnn_transformer import CNNTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


def load_data():
    # uses the files created by data_preprocess.py
    X = np.load("data_processed/X.npy")   # (N, channels, time)
    y = np.load("data_processed/y.npy")   # (N,)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader


def get_class_weights():
    # load labels once
    y = np.load("data_processed/y.npy")   # 0 = baseline, 1 = stress

    classes, counts = np.unique(y, return_counts=True)
    total = y.shape[0]

    weights = []
    for c in [0, 1]:  # 0 = baseline, 1 = stress
        if c in classes:
            freq = counts[classes.tolist().index(c)] / total
            weights.append(1.0 / freq)   # inverse frequency
        else:
            weights.append(0.0)

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()    # normalize
    print("Class weights (baseline, stress):", weights)
    return torch.tensor(weights, dtype=torch.float32)


def train_model():
    loader = load_data()

    # CNN → Transformer model, 2 classes (0/1)
    model = CNNTransformer(
        in_channels=4,
        conv_channels=(32, 64),
        conv_kernel_size=5,
        conv_pool=2,
        d_model=64,
        nhead=2,
        num_transformer_layers=1,
        dim_feedforward=128,
        dropout=0.3,
        num_classes=2,
    ).to(DEVICE)

    # weighted loss to care more about stress class
    class_weights = get_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        all_preds = []
        all_targets = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)      # (batch, 2)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="binary", zero_division=0
        )

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    torch.save(model.state_dict(), "models/cnn_transformer.pth")
    print("Model saved at models/cnn_transformer.pth")


if __name__ == "__main__":
    train_model()
