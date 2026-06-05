import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
    LSTM, Bidirectional, Attention, GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.callbacks import ModelCheckpoint

print("🔥 TRAIN_COMBINED_MODEL.PY IS RUNNING 🔥")

# Load data
X = np.load("data_processed/X.npy")
y = np.load("data_processed/y.npy")

print(f"[INFO] X shape: {X.shape}")
print(f"[INFO] y shape: {y.shape}")

input_layer = Input(shape=X.shape[1:])

# --- CNN Branch ---
cnn = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
cnn = MaxPooling1D(pool_size=2)(cnn)
cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(cnn)
cnn = MaxPooling1D(pool_size=2)(cnn)
cnn = Flatten()(cnn)

# --- BiLSTM Branch ---
bilstm = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
bilstm = GlobalAveragePooling1D()(bilstm)

# --- Attention Branch ---
att_lstm = LSTM(64, return_sequences=True)(input_layer)
att_out = Attention()([att_lstm, att_lstm])
att_out = GlobalAveragePooling1D()(att_out)

# --- Combine branches ---
combined = Concatenate()([cnn, bilstm, att_out])
dense1 = Dense(64, activation='relu')(combined)
drop = Dropout(0.5)(dense1)
output = Dense(1, activation='sigmoid')(drop)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Save best model
checkpoint = ModelCheckpoint(
    "stress_cnn_bilstm_attention.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train model
history = model.fit(
    X, y,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint]
)

print("✅ Combined CNN + BiLSTM + Attention model trained successfully!")
