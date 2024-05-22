# Handwritten-digit-recognition-using-deep-learning
# Overview
This repository contains a deep learning project that recognizes handwritten digits using a Convolutional Neural Network (CNN). The project leverages the popular MNIST dataset to train the model and achieves high accuracy in classifying handwritten digits from 0 to 9.

# Table of Contents
Introduction
Dataset
Model Architecture
Installation
Usage
Results
Contributing
Acknowledgments

# Introduction
Handwritten digit recognition is a fundamental project in the field of computer vision and deep learning. This project uses a Convolutional Neural Network (CNN) to classify images of handwritten digits from the MNIST dataset. The MNIST dataset is a well-known dataset containing 60,000 training images and 10,000 test images of handwritten digits.

# Dataset
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). Each image is labeled with the corresponding digit. The dataset is split into:

Training set: 60,000 images
Test set: 10,000 images

# Model Architecture
The Convolutional Neural Network (CNN) model used in this project consists of the following layers:

Input Layer: 28x28 grayscale images
Convolutional Layer 1: 32 filters, kernel size 3x3, ReLU activation
Max Pooling Layer 1: pool size 2x2
Convolutional Layer 2: 64 filters, kernel size 3x3, ReLU activation
Max Pooling Layer 2: pool size 2x2
Flatten Layer
Dense Layer 1: 128 neurons, ReLU activation
Output Layer: 10 neurons (one for each digit), Softmax activation
# Installation
To run this project, you need to have Python and the following libraries installed:

TensorFlow
Keras
NumPy
Matplotlib

# Acknowledgments
The MNIST database of handwritten digits
TensorFlow and Keras libraries for providing the tools to build and train the model
