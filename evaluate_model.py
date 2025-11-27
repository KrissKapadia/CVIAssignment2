import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. Configuration ---
INFERENCE_DIR = 'Q1/test'
MODEL_PATH = 'best_cat_dog_final_model.h5'
IMAGE_SIZE = (150, 150) 
BATCH_SIZE = 32

# Mapping the model's output index back to class names
class_names = {0: 'Cat', 1: 'Dog'} 

# Check for model existence
if not os.path.exists(MODEL_PATH):
    print(f"Error: Trained model file '{MODEL_PATH}' not found. Please run the training script first.")
    exit()

# --- 2. Load Model ---
print(f"--- 1. Loading Trained Model from {MODEL_PATH} ---")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 3. Prepare Inference Data Generator ---
print(f"\n--- 2. Preparing Inference Data: '{INFERENCE_DIR}' ---")
if not os.path.isdir(INFERENCE_DIR):
    print(f"Error: Inference directory '{INFERENCE_DIR}' not found.")
    exit()

# Data Generator for prediction: only rescaling, NO shuffling
test_datagen = ImageDataGenerator(rescale=1./255)

inference_generator = test_datagen.flow_from_directory(
    INFERENCE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # CRITICAL for matching file names to predictions
)

# Get the true labels and file names
true_labels = inference_generator.classes
filenames = inference_generator.filenames

# --- 4. Generate Predictions ---
print("\n--- 3. Generating Detailed Predictions ---")
# Use the model to predict probabilities for all images
probabilities = model.predict(inference_generator, steps=len(inference_generator))

# Convert probabilities (0 to 1) to binary class predictions (0 or 1)
predicted_classes_binary = (probabilities > 0.5).astype(int).flatten()

# --- 5. Compile Results and Calculate Individual Accuracy ---

results = []
correct_predictions = 0

for i in range(len(filenames)):
    filename = filenames[i]
    true_label_index = true_labels[i]
    predicted_label_index = predicted_classes_binary[i]
    
    # Probability of the PREDICTED class
    if predicted_label_index == 1: # Dog
        confidence = probabilities[i][0]
    else: # Cat
        confidence = 1.0 - probabilities[i][0] # Confidence = 1 - p(Dog)
        
    is_correct = (true_label_index == predicted_label_index)
    
    if is_correct:
        correct_predictions += 1

    results.append({
        'Filename': filename,
        'True Label': class_names[true_label_index],
        'Predicted Label': class_names[predicted_label_index],
        'Confidence': f"{confidence:.4f}",
        'Correct': 'Yes' if is_correct else 'No'
    })

# --- 6. Display Results ---

# Convert the list of results into a Pandas DataFrame for clean printing
df = pd.DataFrame(results)

print("\n=======================================================")
print("           DETAILED INFERENCE RESULTS                   ")
print("=======================================================")
print(df.to_string(index=False))

# --- 7. Print Final Summary ---
overall_accuracy = correct_predictions / len(filenames)

print("\n=======================================================")
print(f"Total Images Evaluated: {len(filenames)}")
print(f"Correct Predictions:    {correct_predictions}")
print(f"Final Test Accuracy:    {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
print("=======================================================")