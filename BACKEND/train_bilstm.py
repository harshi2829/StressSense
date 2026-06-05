import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import pickle

print("🔥 TRAIN_BiLSTM.PY IS RUNNING 🔥")

# Load processed data
X = np.load("data_processed/X.npy")
y = np.load("data_processed/y.npy")

print(f"[INFO] X shape: {X.shape}")
print(f"[INFO] y shape: {y.shape}")

# Build BiLSTM-only model
model = Sequential([
    Input(shape=X.shape[1:]),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Save best model
checkpoint = ModelCheckpoint("bilstm_model.h5", monitor='val_accuracy', save_best_only=True)

# Train model
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
print("✅ BiLSTM-only model trained successfully!")
