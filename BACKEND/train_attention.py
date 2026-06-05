import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint

print("🔥 TRAIN_ATTENTION.PY IS RUNNING 🔥")

# Load data
X = np.load("data_processed/X.npy")
y = np.load("data_processed/y.npy")

print(f"[INFO] X shape: {X.shape}")
print(f"[INFO] y shape: {y.shape}")

# Attention model with proper output shape
input_layer = Input(shape=X.shape[1:])
lstm_out = LSTM(64, return_sequences=True)(input_layer)
attention_out = Attention()([lstm_out, lstm_out])
# Pool over time dimension to get one output per sample
pooled = GlobalAveragePooling1D()(attention_out)
dense1 = Dense(64, activation='relu')(pooled)
drop = Dropout(0.5)(dense1)
output = Dense(1, activation='sigmoid')(drop)  # final output shape: (batch_size, 1)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Save best model
checkpoint = ModelCheckpoint("attention_model.h5", monitor='val_accuracy', save_best_only=True)

# Train model
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
print("✅ Attention-only model trained successfully!")
