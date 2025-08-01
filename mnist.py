# Import all necessary libraries
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
import numpy as np
import os

# load train and test dataset
def load_dataset():
    """
    Loads the MNIST dataset and prepares it for the model.
    - Reshapes images to include a single channel.
    - One-hot encodes the target labels.
    """
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    """
    Normalizes pixel values from integers (0-255) to floats (0.0-1.0).
    """
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

# define cnn model
def define_model():
    """
    Defines and compiles a Convolutional Neural Network (CNN) model for MNIST.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# a. Complete the “evaluate_model” function
def evaluate_model(dataX, dataY, n_folds=5):
    """
    Evaluates a model using k-fold cross-validation.
    A new model is defined and trained for each fold.
    """
    scores, histories = list(), list()
    
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model for this fold
        model = define_model()
        
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print(f'> Accuracy: {acc * 100.0:.3f}%')
        
        # stores scores
        scores.append(acc)
        histories.append(history)
        
    return scores, histories

# Main execution block
if __name__ == '__main__':
    # b. Write a small code to call the functions of “load_dataset”, “prep_pixels” and “evaluate_model”
    
    # Load and prepare the dataset
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    
    # Evaluate the model using k-fold cross-validation
    print("--- Starting K-Fold Cross-Validation ---")
    scores, histories = evaluate_model(trainX, trainY)
    print("--- K-Fold Cross-Validation Finished ---")
    
    # Summarize cross-validation performance
    print(f'Mean Accuracy: {mean(scores) * 100.0:.3f}%')
    print(f'Standard Deviation: {std(scores) * 100.0:.3f}%')
    
    # c. Save the classification model into your local drive
    
    # Define, train, and save a final model on the entire training dataset
    final_model = define_model()
    print("--- Training Final Model on Full Dataset ---")
    final_model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=1)
    
    # Check if the 'models' directory exists, create it if not
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model_filename = 'models/final_model.keras'
    final_model.save(model_filename)
    print(f"Final model saved to '{model_filename}'")
    
    # e. Write a small code to load the saved classification model and predict "sample_image"
    # This assumes that you have a file named 'sample_image.png' in the same directory.
    
    # Define the path to the sample image
    image_path = 'sample_image.png'
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: The image file '{image_path}' was not found.")
        print("Please ensure you have downloaded 'sample_image.png' as instructed in part (d).")
    else:
        print("--- Loading Model and Making a Prediction ---")
        
        # Load the saved model
        try:
            loaded_model = load_model(model_filename)
            print(f"Successfully loaded model from '{model_filename}'")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()
            
        # Load and prepare the sample image
        img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
        img_array = img_to_array(img)
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
        img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
        
        # Make a prediction
        prediction = loaded_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        
        print(f"The predicted digit is: {predicted_class[0]}")
