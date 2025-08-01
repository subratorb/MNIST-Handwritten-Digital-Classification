<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST Handwritten Digit Classifier README</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #1a202c;
            color: #e2e8f0;
        }
        .container {
            max-width: 800px;
        }
        pre {
            background-color: #2d3748;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            color: #cbd5e0;
        }
        code {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            background-color: #4a5568;
            padding: 0.2rem 0.4rem;
            border-radius: 0.3rem;
            font-weight: 500;
        }
    </style>
</head>
<body class="p-8">
    <div class="container mx-auto bg-gray-800 p-8 rounded-xl shadow-lg">

        <!-- Header -->
        <header class="mb-8 border-b-2 border-gray-600 pb-4">
            <h1 class="text-4xl font-bold mb-2 flex items-center">
                🤖 MNIST Handwritten Digit Classifier
            </h1>
            <p class="text-gray-400 text-lg">
                A project to classify handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN).
            </p>
        </header>
        
        <!-- Features Section -->
        <section class="mb-8">
            <h2 class="text-3xl font-semibold mb-4">✨ Features</h2>
            <ul class="list-none space-y-3 pl-0">
                <li class="flex items-start">
                    <span class="mr-3 text-2xl">✔️</span>
                    <div>
                        <strong class="text-xl">Data Preparation</strong>: Loads, reshapes, and normalizes the MNIST dataset for model consumption.
                    </div>
                </li>
                <li class="flex items-start">
                    <span class="mr-3 text-2xl">✔️</span>
                    <div>
                        <strong class="text-xl">Robust Evaluation</strong>: Utilizes **5-fold cross-validation** to measure model performance accurately.
                    </div>
                </li>
                <li class="flex items-start">
                    <span class="mr-3 text-2xl">✔️</span>
                    <div>
                        <strong class="text-xl">Model Persistence</strong>: Saves the final trained model for future use without retraining.
                    </div>
                </li>
                <li class="flex items-start">
                    <span class="mr-3 text-2xl">✔️</span>
                    <div>
                        <strong class="text-xl">Prediction</strong>: Demonstrates how to load a saved model to predict a new digit from an image.
                    </div>
                </li>
            </ul>
        </section>

        <!-- Getting Started Section -->
        <section class="mb-8">
            <h2 class="text-3xl font-semibold mb-4">🚀 Getting Started</h2>
            
            <h3 class="text-2xl font-medium mb-2">Prerequisites</h3>
            <p class="text-gray-300 mb-4">You'll need Python 3 and the following libraries. Install them with `pip`:</p>
            <pre><code>pip install tensorflow numpy scikit-learn</code></pre>
            
            <h3 class="text-2xl font-medium mt-6 mb-2">Usage</h3>
            <ol class="list-decimal list-inside space-y-3 text-gray-300">
                <li>
                    <strong>Create the file</strong>: Save the provided Python script as <code>mnist_classifier.py</code>.
                </li>
                <li>
                    <strong>Add your image</strong>: Place a grayscale image of a digit, named <code>sample_image.png</code>, in the same directory.
                </li>
                <li>
                    <strong>Run the script</strong>: Execute the script from your terminal.
                </li>
            </ol>
            <pre class="mt-4"><code>python mnist_classifier.py</code></pre>
            <p class="mt-4 text-gray-300">
                The script will:
            </p>
            <ul class="list-disc list-inside mt-2 space-y-2 text-gray-300">
                <li>Perform cross-validation and print the accuracy for each fold.</li>
                <li>Train a final model on the full dataset and save it as <code>final_model.keras</code>.</li>
                <li>Load the saved model and predict the digit in your <code>sample_image.png</code>.</li>
            </ul>
        </section>

        <!-- Code Highlights Section -->
        <section class="mb-8">
            <h2 class="text-3xl font-semibold mb-4">🧠 Code Highlights</h2>
            <ul class="list-disc list-inside space-y-2 text-gray-300">
                <li><code>load_dataset()</code>: Loads and preprocesses the MNIST data.</li>
                <li><code>prep_pixels()</code>: Normalizes image pixel values.</li>
                <li><code>define_model()</code>: Defines the CNN architecture.</li>
                <li><code>evaluate_model()</code>: Implements k-fold cross-validation.</li>
            </ul>
        </section>

        <!-- License Section -->
        <section>
            <h2 class="text-3xl font-semibold mb-4">📄 License</h2>
            <p class="text-gray-300">
                This project is open-sourced under the 
                <a href="https://opensource.org/licenses/MIT" class="text-blue-400 hover:text-blue-300 transition-colors duration-200">MIT license</a>.
            </p>
        </section>

    </div>
</body>
</html>
