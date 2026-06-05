import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

print("🔥 TRAIN_CNN.PY IS RUNNING 🔥")

# Load data
X = np.load("data_processed/X.npy")
y = np.load("data_processed/y.npy")

print(f"[INFO] X shape: {X.shape}")
print(f"[INFO] y shape: {y.shape}")

# Shallow CNN model
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=X.shape[1:]),
    # Optional: remove or reduce pooling since input length is small
    Conv1D(64, kernel_size=2, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Save best model
checkpoint = ModelCheckpoint("cnn_model.h5", monitor='val_accuracy', save_best_only=True)

# Train model
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
print("✅ CNN-only model trained successfully!")
