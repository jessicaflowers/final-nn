[![BuildStatus](https://github.com/jessicaflowers/final-nn/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/jessicaflowers/final-nn/actions/workflows/test.yml)

# Final project: neural network

## Overview

In this assignment, you will implement a neural network class from (almost) scratch. You will then apply your class to create both:

**(1) a simple 64x16x64 autoencoder.**

**(2) a classifier for transcription factor binding sites.**

You will begin by finishing the API for generating fully connected neural networks from scratch. You will then make Jupyter Notebooks where you create, train, and test your autoencoder and classifier.

## Step 1: finish the neural network API

### For steps 2 and 3

* Finish all methods with a `pass` statement in the `NeuralNetwork` class in the `nn.py` file.

### For step 3

* Finish the `sample_seqs` function in the `preprocess.py` file.
* Finish the `one_hot_encode_seqs` function in the `preprocess.py` file.

## Step 2: make your autoencoder

### Background

An autoencoder is a neural network that takes an input, encodes it into a lower-dimensional latent space through "encoding" layers, and then attempts to reconstruct the original input using "decoding" layers. Autoencoders are often used for dimensionality reduction.

### Your task

You will train a 64x16x64 autoencoder on the [digits](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset) dataset. All of the following work should be done in a Jupyter Notebook.

### To-do

* Load the digits dataset through sklearn using <code><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html">sklearn.datasets.load_digits()</a></code>.
* Split the data into training and validation sets.
* Generate an instance of your `NeuralNetwork` class with a 64x16x64 autoencoder architecture.
* Train your autoencoder on the training data.
* Plot your training and validation loss by epoch.
* Quantify your average reconstruction error over the validation set.
* Explain why you chose the hyperparameter values you did.

## Step 3: make your classifier

### Background

Transcription factors are proteins that bind DNA at promoters to drive gene expression. Most preferentially bind to specific sequences while ignoring others. Traditional methods to determine these sequences (called motifs) have assumed that binding sites in the genome are all independent. However, in some cases people have identified motifs where positional interdependencies exist.

### Your task

You will implement a multi-layer fully connected neural network using your `NeuralNetwork` class to predict whether a short DNA sequence is a binding site for the yeast transcription factor Rap1. The training data is incredibly imbalanced, with way fewer positive sequences than negative sequences, so you will implement a sampling scheme to ensure that class imbalance does not affect training. As in step 2, all of the following work should be done in a Jupyter Notebook.

### To-do

* Use the `read_text_file` function from `io.py` to read in the 137 positive Rap1 motif examples.
* Use the `read_fasta_file` function from `io.py` to read in all the negative examples. Note that these sequences are much longer than the positive sequences, so you will need to process them to the same length.
* Balance your classes using your `sample_seq` function and explain why you chose the sampling scheme you did.
* One-hot encode the data using your `one_hot_encode_seqs` function.
* Split the data into training and validation sets.
* Generate an instance of your `NeuralNetwork` class with an appropriate architecture.
* Train your neural network on the training data.
* Plot your training and validation loss by epoch.
* Report the accuracy of your classifier on your validation dataset.
* Explain your choice of loss function and hyperparameters.

## Grading (50 points)

### Neural network implementation (15 points)

* Proper implementation of methods in `NeuralNetwork` class (13 points)
* Proper implementation of `sample_seqs` function (1 point)
* Proper implementation of `one_hot_encode_seqs` function (1 point)

### Autoencoder (10 points)

* Read in data and generate training and validation sets (2 points)
* Successfully train your autoencoder (4 points)
* Plots of training and validation loss (2 points)
* Quantification of reconstruction error (1 point)
* Explanation of hyperparameters (1 point)

### Classifier (15 points)

* Correctly read in all data (2 points)
* Explanation of your sampling scheme (2 points)
* Proper generation of a training set and a validation set (2 point)
* Successfully train your classifier (4 points)
* Plots of training and validation loss (2 points)
* Reporting validation accuracy of the classifier (1 point)
* Explanation of loss function and hyperparameters (2 points)

### Testing (7 points)

Proper unit tests for:

* `_single_forward` method (1 point)
* `forward` method (1 point)
* `_single_backprop` method (1 point)
* `predict` method (1 point)
* `binary_cross_entropy` method (0.5 points)
* `binary_cross_entropy_backprop` method (0.5 points)
* `mean_squared_error` method (0.5 points)
* `mean_squared_error_backprop` method (0.5 points)
* `sample_seqs` function (0.5 points)
* `one_hot_encode_seqs` function (0.5 points)

### Packaging (3 points)

* Installable module (1 point)
* GitHub Actions (installing + testing) (2 points)

### Submission
Please submit a link to your final project repo [here](https://forms.gle/9xWdSinubVTYTwL2A)
