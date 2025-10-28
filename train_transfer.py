# STEP 0: Import libraries
import tensorflow as tf
import os
import zipfile
import logging
import matplotlib.pyplot as plt

# --- NEW IMPORTS FOR TRANSFER LEARNING ---
from tensorflow.keras.applications import MobileNetV2 # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Model # type: ignore
# --- End of new imports ---

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Configure logging (no changes)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

print("--- Start script: Transfer Learning (MobileNetV2) ---")

# --- STEP 1: DOWNLOAD DATA (no changes) ---
print("\n[Step 1/6] Downloading and unpacking data...")
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_file_name = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=url, extract=False)
base_dir = os.path.join(os.path.dirname(zip_file_name), 'cats_and_dogs_filtered')
if not os.path.exists(base_dir):
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_file_name))
    print(f"Data unpacked to: {base_dir}")
else:
    print(f"Data folder already exists: {base_dir}")
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# --- STEP 2: PREPARE DATA GENERATORS ---
print("\n[Step 2/6] Preparing data generators...")

# Define constants
IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 20
EPOCHS = 20 # We can use fewer epochs, it learns faster

# --- KEY CHANGE: PRE-PROCESSING ---
# MobileNetV2 expects pixels in the range [-1, 1], not [0, 1].
# We will use its built-in `preprocess_input` function instead of `rescale=1./255`.

train_image_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input, # <--- Use the dedicated function
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation generator must use the same function (but no augmentation)
validation_image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load data (no changes)
try:
    train_data_gen = train_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE, directory=train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE, directory=validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')
    
    print("Data generators (with preprocess_input) created.")
    print(f"Class labels: {train_data_gen.class_indices}")
except Exception as e:
    print(f"ERROR during generator creation: {e}")
    exit()

# --- STEP 3: BUILD THE MODEL (Completely new code) ---
print("\n[Step 3/6] Building the model (Transfer Learning)...")

# 1. Load the expert "brain" (MobileNetV2)
# `weights='imagenet'` -> load knowledge from millions of photos
# `include_top=False` -> cut off the final layer (the one for 1000 classes)
# `input_shape` -> tell the model our images are 150x150x3
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

# 2. "Freeze" the expert's brain
# Tell it: "Don't re-learn anything in these layers, they are perfect"
base_model.trainable = False
print(f"Base model {base_model.name} loaded. Its layers have been FROZEN.")

# 3. Build our new, custom "head" (classifier)
inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = base_model(inputs, training=False) # 'training=False' is important for a frozen model
x = GlobalAveragePooling2D()(x) # Flattens the output
x = Dropout(0.2)(x) # Add Dropout for regularization
outputs = Dense(1, activation='sigmoid')(x) # Our old output layer (cat/dog)

# 4. Combine the frozen "brain" (base_model) with our "new head" (outputs)
model = Model(inputs, outputs)

print("New 'head' has been added to the base model.")
model.summary()

# --- STEP 4: COMPILE AND TRAIN THE MODEL ---
print("\n[Step 4/6] Compiling and training the model...")
print(f"Training will run for {EPOCHS} epochs.")

# We use the same low learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Calculate steps (no changes)
steps_per_epoch = train_data_gen.n // BATCH_SIZE
validation_steps = val_data_gen.n // BATCH_SIZE

# Start training!
history = model.fit(
    train_data_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=validation_steps
)
print("Training finished.")

# --- STEP 5: SAVE THE MODEL ---
print("\n[Step 5/6] Saving the trained model...")
# Let's save it under a new name
model.save('dogs_vs_cats_TRANSFER_model.keras')
print("Model was saved to 'dogs_vs_cats_TRANSFER_model.keras'")

# --- STEP 6: EVALUATE THE MODEL (PLOTS) (no changes) ---
print("\n[Step 6/6] Plotting accuracy and loss graphs...")
# (This code is identical to the previous script)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.savefig('training_history_plot_TRANSFER.png')
plt.show()

print("\n--- Script finished ---")