# Dog vs. Cat Image Classifier 🐶🐱

This project is an image classifier built with Python, TensorFlow, and Keras. It demonstrates two approaches to building a Convolutional Neural Network (CNN) to distinguish between dogs and cats.

## Project Structure

* `train.py`: A simple CNN model built from scratch (achieves ~75% accuracy).
* `train_transfer.py`: A professional model using **Transfer Learning** (with MobileNetV2). This model achieves **>95% accuracy**.
* `predict.py`: A script to run predictions on new images using the best (transfer learning) model.
* `requirements.txt`: All necessary Python libraries.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TWOJA_NAZWA/NAZWA_REPOZYTORIUM.git](https://github.com/TWOJA_NAZWA/NAZWA_REPOZYTORIUM.git)
    cd NAZWA_REPOZYTORIUM
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate  (on Windows)
    pip install -r requirements.txt
    ```

3.  **Train the model:**
    To get the high-accuracy model, you must run the transfer learning script. This will generate the `dogs_vs_cats_TRANSFER_model.keras` file (which is ignored by Git).
    ```bash
    python train_transfer.py
    ```

4.  **Run predictions:**
    Pass the path to your own image to the `predict.py` script.
    ```bash
    python predict.py "C:\Path\To\Your\Image.jpg"
    ```