#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    # shape (10, 5)
    # each row signifies an input node
    # each column is a different target node
    # W1[2,3] is the connection from the 3rd input to the 4th target
    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = np.matmul(data, W1) + b1   # output is shape (M, H) (i.e. M, 5)
    a1 = sigmoid(z1)                # output is shape (M, H)

    z2 = np.matmul(a1, W2) + b2     # output is shape (M, Dy)
    a2 = sigmoid(z2)                # output is shape (M, Dy)

    y_hat = softmax(a2)             # output is shape (M, Dy)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    cost = (-1)*(labels*np.log(y_hat)).sum()
    grad_cost = y_hat - labels      # output is shape (M, Dy)

    error_output = grad_cost * sigmoid_grad(z2)     # output is shape (M, Dy)
    gradb2 = error_output

    error_hidden_layer = np.matmul(error_output, W2.transpose()) * sigmoid_grad(z1) # output is shape (M, H)
    gradb1 = error_hidden_layer

    gradW1 = np.matmul(data.transpose(), error_hidden_layer)  # output is shape (M, H)
    gradW2 = np.matmul(a1.transpose(), error_output)

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE
    return

if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
