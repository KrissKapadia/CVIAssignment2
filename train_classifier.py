import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- 1. Configuration ---
# Directory containing your training data (assumes a 'Cat' and 'Dog' subdirectory)
TRAIN_DIR = 'Q1/train' 
IMAGE_SIZE = (150, 150) # All images will be resized to this standard for the model
BATCH_SIZE = 32
INITIAL_EPOCHS = 30 # Epochs for the initial training (frozen base)
FINE_TUNE_EPOCHS = 10 # Additional epochs for fine-tuning
INITIAL_LR = 1e-4 # Learning rate for initial phase
FINE_TUNE_LR = 1e-5 # Very small learning rate for fine-tuning
CHECKPOINT_PATH = 'best_cat_dog_initial_model.h5'
FINAL_MODEL_PATH = 'best_cat_dog_final_model.h5'

# Check if the directory exists
if not os.path.isdir(TRAIN_DIR):
    print(f"Error: Training directory '{TRAIN_DIR}' not found.")
    print("Please ensure you have created this folder and populated it with 'Cat' and 'Dog' subfolders.")
    exit()

# --- 2. Data Preparation and Augmentation ---
print("--- 2. Setting up Data Generators ---")

# Data Augmentation for Training Set and Normalization for all
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 # Use 20% of Train_Data for validation
)

# Training Data Generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# Validation Data Generator
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print(f"Class mapping: {train_generator.class_indices} (e.g., Cat=0, Dog=1)")

# --- 3. Model Construction (Transfer Learning) ---
print("\n--- 3. Building VGG16 Transfer Learning Model ---")

# Load Pre-trained Base Model (VGG16)
conv_base = VGG16(
    weights='imagenet',          # Load weights trained on ImageNet
    include_top=False,           # Exclude the default classification layer
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3) # Match our input size
)

# Freeze the convolutional base weights
conv_base.trainable = False 

# Define the Custom Classifier Head
model = Sequential([
    conv_base,
    GlobalAveragePooling2D(), # Efficiently reduces spatial dimensions
    Dense(256, activation='relu'),
    Dropout(0.5), # Regularization
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Compile the Model
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 4. Training: Phase 1 (Feature Extraction) ---
print("\n--- 4. Starting Initial Training (Frozen Base) ---")

# Define Callbacks
model_checkpoint = ModelCheckpoint(
    CHECKPOINT_PATH, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=5, 
    mode='max', 
    verbose=1
)

# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[model_checkpoint, early_stopping]
)

# Load the best weights from Phase 1
model.load_weights(CHECKPOINT_PATH)
initial_epoch_count = len(history.epoch)

# --- 5. Training: Phase 2 (Fine-Tuning) ---
print("\n--- 5. Starting Fine-Tuning of Top Layers ---")

# Unfreeze the top layers of the convolutional base
conv_base.trainable = True

# Freeze all layers except the last few blocks (e.g., the last three convolutional layers in VGG16)
for layer in conv_base.layers[:-3]:
    layer.trainable = False
    
# Recompile the model with a very low learning rate for stability
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Continue training
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=initial_epoch_count + FINE_TUNE_EPOCHS, # Total number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    initial_epoch=initial_epoch_count # Start from where previous training left off
)

# --- 6. Final Save ---
model.save(FINAL_MODEL_PATH)
print(f"\n Training complete! Final fine-tuned model saved as: {FINAL_MODEL_PATH}")

# Clean up checkpoint file
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)