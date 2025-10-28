import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]
import sys
import os
import logging

# --- NEW IMPORT: preprocessing function for MobileNetV2 ---
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # pyright: ignore[reportMissingImports]

# --- Configuration (hides unnecessary TensorFlow logs) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

# --- Constants ---
IMG_WIDTH = 150
IMG_HEIGHT = 150

# --- CHANGE 1: Path to the new model ---
MODEL_PATH = 'dogs_vs_cats_TRANSFER_model.keras' # <--- Points to the new model

def predict_image(image_path):
    # 1. Load the saved model
    print(f"Loading model from file: {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except (IOError, OSError):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Make sure you have run 'train_transfer.py' at least once.")
        return

    # 2. Load and prepare the image
    try:
        # Load the image and resize it to 150x150
        img = image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        
        # Add a batch dimension (from 150,150,3 to 1,150,150,3)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # --- CHANGE 2: Use the correct preprocessing function ---
        # Instead of 'img_preprocessed = img_batch / 255.0'
        img_preprocessed = preprocess_input(img_batch) # <--- Function for MobileNetV2
        
    except Exception as e:
        print(f"ERROR: Could not load or process image: {e}")
        return

    # 3. Prediction (use the model)
    print("Model is analyzing the image...")
    prediction = model.predict(img_preprocessed)
    
    # The output is a score between 0.0 (cat) and 1.0 (dog)
    score = prediction[0][0]

    # 4. Interpret the result
    print("\n--- ANALYSIS RESULT ---")
    print(f"Raw model output: {score:.4f}")
    
    if score > 0.5:
        print(f"Verdict: This is a DOG 🐶 (Confidence: {score*100:.2f}%)")
    else:
        print(f"Verdict: This is a CAT 🐱 (Confidence: {(1-score)*100:.2f}%)")

# --- This part runs the script from the terminal ---
if __name__ == "__main__":
    # Check if the user provided exactly one argument (the file path)
    if len(sys.argv) != 2:
        print("ERROR: Invalid script usage.")
        print("You must provide a path to an image as an argument.")
        print("\nExample usage:")
        # Updated the example path to be generic
        print('python predict.py "C:\\Path\\To\\Your\\Image.jpg"')
    else:
        # Get the file path from the command line argument
        file_path = sys.argv[1]
        predict_image(file_path)