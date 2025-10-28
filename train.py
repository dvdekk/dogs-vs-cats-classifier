# STEP 0: Import libraries and basic configuration
import tensorflow as tf
import os
import zipfile
import logging
import matplotlib.pyplot as plt

# Disable informational logs from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

print("--- Start script: Simple CNN Classifier ---")

# --- STEP 1: DOWNLOAD AND PREPARE DATA ---
print("\n[Step 1/6] Downloading and unpacking data...")

url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_file_name = tf.keras.utils.get_file(
    'cats_and_dogs_filtered.zip',
    origin=url,
    extract=False
)

# Path where the data will be unpacked
base_dir = os.path.join(os.path.dirname(zip_file_name), 'cats_and_dogs_filtered')

# Unpack the file (only if the folder doesn't exist yet)
if not os.path.exists(base_dir):
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_file_name))
    print(f"Data unpacked to: {base_dir}")
else:
    print(f"Data folder already exists: {base_dir}")

# Define paths to training and validation directories
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# --- STEP 2: PREPARE DATA GENERATORS ---
print("\n[Step 2/6] Preparing data generators...")

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore # type: ignore

# Define constants for our model
IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 20
EPOCHS = 30     # We set this to 30 after our first failed attempt

# Generator for TRAINING data (with augmentation)
train_image_generator = ImageDataGenerator(
    rescale=1./255, # Pixel scaling (0-255 -> 0-1)
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generator for VALIDATION data (only scaling!)
validation_image_generator = ImageDataGenerator(rescale=1./255)

# Load data from folders
try:
    train_data_gen = train_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary' # 'binary' = two classes (cat/dog)
    )

    val_data_gen = validation_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary'
    )
    print("Data generators created successfully.")
    print(f"Class labels: {train_data_gen.class_indices}") # Will print {'cats': 0, 'dogs': 1}
except Exception as e:
    print(f"ERROR during generator creation: {e}")
    exit()

# --- STEP 3: BUILD THE CNN MODEL ---
print("\n[Step 3/6] Building the CNN model...")

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dropout(0.5), # To prevent overfitting
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid') # 1 neuron, for binary classification (0 or 1)
])

print("Model architecture has been built:")
model.summary()

# --- STEP 4: COMPILE AND TRAIN THE MODEL ---
print("\n[Step 4/6] Compiling and training the model...")
print(f"Training will run for {EPOCHS} epochs. This might take a while on a CPU...")

from tensorflow.keras.optimizers import Adam # type: ignore

# We use a smaller learning rate for stable training
model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss='binary_crossentropy', # Loss function for binary classification
    metrics=['accuracy']
)

# Calculate steps per epoch
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

# Save the entire model (architecture + weights) to one file
model.save('dogs_vs_cats_model.keras')
print("Model was saved to 'dogs_vs_cats_model.keras'")


# --- STEP 6: EVALUATE THE MODEL (PLOTS) ---
print("\n[Step 6/6] Plotting accuracy and loss graphs...")

# Get data from training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

# Create plots
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Save the plot to a file and show it
plt.savefig('training_history_plot.png')
print("Plots saved to 'training_history_plot.png'")
plt.show()

print("\n--- Script finished ---")