🤖 MNIST Handwritten Digit Classifier
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the famous MNIST dataset. It serves as a practical example for understanding key machine learning concepts like data preprocessing, model building, and k-fold cross-validation.

✨ Features
Data Preparation: Automatically loads, reshapes, and normalizes the MNIST dataset.

Robust Evaluation: Uses 5-fold cross-validation to ensure the model's accuracy is reliable.

Model Management: Trains a final model on the full dataset and saves it to a .keras file.

Prediction: Includes a script to load the saved model and predict a new, unseen digit from a sample image.

🚀 Getting Started
Prerequisites
You'll need Python 3 and the following libraries. You can install them with pip:

Bash

pip install tensorflow numpy scikit-learn
Usage
Save the Script: Save the Python code as mnist_classifier.py in your project directory.

Get the Sample Image: Make sure you have a grayscale image file of a handwritten digit, named sample_image.png, in the same directory.

Run the Script: Execute the script from your terminal.

Bash

python mnist_classifier.py
The script will print the cross-validation scores, train a final model, save it, and then provide a prediction for the sample_image.png.

📂 Project Structure
mnist_classifier.py: The main Python script containing all functions and the execution logic.

models/: A folder created by the script to store the trained final_model.keras.

sample_image.png: The sample image used for prediction.
