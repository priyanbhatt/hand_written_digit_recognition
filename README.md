**Handwritten Digit Recognition with FCN, CNN & Random Forest**

Project Overview

This repository contains an end-to-end comparison of three approaches for classifying handwritten digits on the MNIST dataset:

Fully-Connected Neural Network (FCN) using TensorFlow/Keras as a baseline.

Convolutional Neural Network (CNN) using TensorFlow/Keras for spatial feature learning.

Random Forest classifier using scikit-learn as a classical machine-learning counterpart.

Each model is trained, evaluated, and compared based on test-set accuracy.

Dataset & Preprocessing

Dataset Size: 20,000 training images and 10,000 test images of handwritten digits, each of size 28×28 pixels in grayscale.

Preprocessing Steps:

Reshape: Images reshaped to include channel dimension (28, 28, 1) for CNN inputs.

Normalization: Pixel values scaled from [0, 255] to [0, 1] by dividing by 255.0.

Train/Test Split: Original MNIST split used directly; no additional hold-out needed since test set provided.

Model Architectures and Explanations

**Fully-Connected Network (FCN)**

Architecture: Input (784) → Dense (128 units, ReLU) → Dense (64 units, ReLU) → Dense (10 units, linear)

![image](https://github.com/user-attachments/assets/324a4356-8d99-4cf1-8ee2-988166143513)


Test Accuracy: 96.55%

Description: A simple multilayer perceptron that flattens the input images and learns global patterns. Fast to train but less effective at capturing spatial hierarchies.

**Convolutional Neural Network (LeNet-style CNN)**

Architecture:

Conv2D (8 filters, 3×3 kernel, ReLU)

MaxPool2D (2×2)

Dropout(0.25)

Conv2D (16 filters, 3×3 kernel, ReLU)

MaxPool2D (2×2)

Dropout(0.25)

Flatten

Dense (256 units, ReLU)

Dense (256 units, ReLU)

Dropout(0.5)

Dense (10 units, Softmax)

![image](https://github.com/user-attachments/assets/21bcad31-cfa3-472d-bfb9-9251b81b4c40)


Test Accuracy: 98.74%

Description: Leverages convolution and pooling to hierarchically extract local features, resulting in superior performance on image data.

**Random Forest (Flattened Inputs)**

Architecture: Flatten (784 input features) → Random Forest (100 estimators)

Test Accuracy: 96.04%

Description: A classical ensemble of decision trees trained on flattened pixel vectors. No GPU required but relies on manual feature flattening and hyperparameter tuning.

Results Comparison

The following table summarizes the test-set accuracies achieved by each model:

Model

Test Accuracy

Fully-Connected Network (FCN)

96.55%

Convolutional Neural Network (CNN)

98.74%

Random Forest (Flattened Inputs)

96.04%

