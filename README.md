# four_or_nine-digit-classifier-

A neural network built completely from scratch using Python and NumPy to classify handwritten digits from the MNIST dataset.

Features
Built without machine learning frameworks
Uses forward propagation and backpropagation
Trained on the MNIST handwritten digit dataset
Supports custom digit drawing and prediction
Model weights can be saved and loaded
How It Works

The neural network learns to recognise handwritten digits by adjusting its weights through training.

Training Process
Load and preprocess the MNIST dataset
Flatten images into 784-pixel input vectors
Perform a forward pass through the network
Calculate prediction error
Use backpropagation to update weights
Repeat for multiple epochs
Network Architecture

Input Layer:

784 neurons (28×28 pixels)

Hidden Layer:

32 neurons
ReLU activation

Output Layer:

Sigmoid activation for binary classification
