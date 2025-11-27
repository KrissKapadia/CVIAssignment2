import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
TEST_FILE_PATH = 'Q2/mnist_test.csv'
MODEL_LOAD_PATH = 'mnist_cnn_model.h5'
IMG_ROWS, IMG_COLS = 28, 28
NUM_CLASSES = 10

# --- 1. Load Model ---
try:
    print(f"Loading model from: {MODEL_LOAD_PATH}")
    model = load_model(MODEL_LOAD_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_LOAD_PATH}. Run 'mnist_train.py' first.")
    exit()

# --- 2. Load and Prepare Test Data ---
try:
    print(f"Loading test data from: {TEST_FILE_PATH}")
    test_data = pd.read_csv(TEST_FILE_PATH, header=None)
except FileNotFoundError:
    print(f"Error: Test file not found at {TEST_FILE_PATH}. Check file path.")
    exit()

# Separate labels and pixel features
y_test_raw = test_data.iloc[:, 0].values
X_test_raw = test_data.iloc[:, 1:].values

# Reshape features: Flattened vector (784) -> Image (28, 28, 1)
X_test = X_test_raw.reshape(X_test_raw.shape[0], IMG_ROWS, IMG_COLS, 1).astype('float32')

# Normalize pixel values to [0, 1]
X_test /= 255

# Convert labels to one-hot encoding
y_test = to_categorical(y_test_raw, NUM_CLASSES)

print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# --- 3. Evaluate the Model ---
print("\nEvaluating model performance on test data...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

# --- 4. Report Results ---
print("\n==============================================")
print(f" Final Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"   Required Accuracy:   90.00%")
print(f"   Test Loss:           {loss:.4f}")
print("==============================================")