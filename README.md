# MNIST-Handwritten-Digital-Classification
The completed Python script for the MNIST handwritten digit classification problem. The evaluate_model function, adding the main execution logic to train and save a final model, and providing a final section to load that model and make a prediction on a sample image.

# Let's dive into the project.
Here is a breakdown of what happens in that code:

Imports: This section brings in all the necessary Python libraries.

numpy is used for numerical operations, especially with arrays.

``` bash
sklearn.model_selection.KFold ``` is for splitting the data into training and testing sets for cross-validation.

``` bash tensorflow.keras ``` is the deep learning framework used to build and train the neural network. The specific imports are for loading the MNIST dataset, preparing the labels (to_categorical), and defining the different layers of the CNN (Sequential, Conv2D, MaxPooling2D, etc.).

os is a standard library for interacting with the operating system, used later in the script to handle file paths.


``` bash load_dataset() ``` function: This is the first step of data preparation. It performs two key tasks:

It loads the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

It reshapes the images to add a channel dimension. A typical grayscale image is 2D (height x width), but a CNN model expects a 3D input (height x width x channels). For MNIST, a single channel is added to represent grayscale images. The function also converts the integer labels (0-9) into a one-hot encoded format (e.g., the digit 7 becomes [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), which is required for the categorical cross-entropy loss function used in the model.

prep_pixels() function: This is the second step of data preparation. It normalizes the pixel values of the images. The original pixel values are integers ranging from 0 to 255. By dividing them by 255.0, the function scales them to a floating-point range of 0.0 to 1.0. This normalization is a common and important step in deep learning, as it helps the model's training process converge faster and more effectively.

define_model() function: This function defines the architecture of the Convolutional Neural Network (CNN).

Sequential() creates a linear stack of layers.

Conv2D(32, (3, 3), ...) is the first layer. It's a 2D convolutional layer with 32 filters, each of size 3×3. It extracts features from the input images using a 'relu' activation function.

MaxPooling2D((2, 2)) is a down-sampling layer that reduces the spatial dimensions of the feature maps, which helps reduce the number of parameters and computational cost.

The Flatten() and Dense() layers that follow are part of the "fully connected" portion of the network, which performs the final classification based on the features extracted by the convolutional layers.

# Result
<img width="1142" height="618" alt="image" src="https://github.com/user-attachments/assets/b2519a89-5158-4b97-93d8-60719f0519e1" />
