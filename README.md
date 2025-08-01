# 🤖 MNIST Handwritten Digit Classifier
A project to classify handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN).
## ✨ Features
Data Preparation: Loads, reshapes, and normalizes the MNIST dataset for model consumption.

Robust Evaluation: Utilizes 5-fold cross-validation to measure model performance accurately.

Model Persistence: Saves the final trained model for future use without retraining.

Prediction: Demonstrates how to load a saved model to predict a new digit from an image.

## 🚀 Getting Started
Prerequisites

You'll need Python 3 and the following libraries. Install them with pip:

```bash
pip install tensorflow numpy scikit-learn
```

Usage
Create the file: Save the provided Python script as _mnist_classifier.py._

Add your image: Place a grayscale image of a digit, named sample_image.png, in the same directory.

Run the script: Execute the script from your terminal.

python mnist_classifier.py

The script will:

Perform cross-validation and print the accuracy for each fold.

Train a final model on the full dataset and save it as final_model.keras.

Load the saved model and predict the digit in your sample_image.png.

🧠 Code Highlights
```bash
load_dataset(): Loads and preprocesses the MNIST data.

prep_pixels(): Normalizes image pixel values.

define_model(): Defines the CNN architecture.

evaluate_model(): Implements k-fold cross-validation.
```

📄 License
This project is open-sourced under the MIT license.
