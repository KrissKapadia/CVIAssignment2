import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
TRAIN_FILE_PATH = 'Q2/mnist_train.csv'
MODEL_SAVE_PATH = 'mnist_cnn_model.h5'
IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NUM_CLASSES = 10
EPOCHS = 10
BATCH_SIZE = 128

# --- 1. Load and Prepare Data ---
try:
    print(f"Loading training data from: {TRAIN_FILE_PATH}")
    train_data = pd.read_csv(TRAIN_FILE_PATH, header=None)
except FileNotFoundError:
    print(f"Error: Training file not found at {TRAIN_FILE_PATH}. Check file path.")
    exit()

# Separate labels (first column) and pixel features
y_train_raw = train_data.iloc[:, 0].values
X_train_raw = train_data.iloc[:, 1:].values

# Reshape features: Flattened vector (784) -> Image (28, 28, 1)
X_train = X_train_raw.reshape(X_train_raw.shape[0], IMG_ROWS, IMG_COLS, 1).astype('float32')

# Normalize pixel values to [0, 1]
X_train /= 255

# Convert labels to one-hot encoding (e.g., 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
y_train = to_categorical(y_train_raw, NUM_CLASSES)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")


# --- 2. Define the CNN Model ---
print("\nDefining CNN Model...")
# A simple, effective CNN architecture for MNIST
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Regularization to prevent overfitting
    Dense(NUM_CLASSES, activation='softmax') # Output layer for 10 classes
])

# --- 3. Compile and Train ---
print("\nStarting model training...")
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model (using 10% of training data for validation during training)
history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_split=0.1)

# --- 4. Save the Model ---
model.save(MODEL_SAVE_PATH)
print(f"\n Training Complete. Final model saved to: {MODEL_SAVE_PATH}")