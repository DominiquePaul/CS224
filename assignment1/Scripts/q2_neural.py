#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    layer1 = sigmoid(np.dot(X, W1) + b1)
    y_ = softmax(np.dot(layer1,W2) + b2)

    print("Shape of X: {}".format(X.shape))
    print("Shape of W1: {}".format(W1.shape))
    print("Shape of b1: {}".format(b1.shape))
    print("Shape of layer1: {}".format(layer1.shape))
    print("Shape of W2: {}".format(W2.shape))
    print("Shape of b2: {}".format(b2.shape))
    print("Shape of y_: {}".format(y_.shape))
    print("\n")
    ### END YOUR CODE


    ### YOUR CODE HERE: backward propagation
    cost = np.sum(-np.multiply(labels, np.log(y_))) / X.shape[0]

    grad_pre_l2 = np.subtract(y_, labels) / X.shape[0]
    gradb2 = np.sum(grad_pre_l2,axis=0)
    gradW2 = np.dot(layer1.T,grad_pre_l2)

    dh = np.dot(grad_pre_l2, W2.T)
    grad_pre_l1 = sigmoid_grad(dh)
    grad_pre_l1 = sigmoid_grad(layer1) * dh

    gradb1 = np.sum(grad_pre_l1, axis=0)
    gradW1 = np.dot(X.T,grad_pre_l1)

    print("Shape of Grad_pre_l2:    {}".format(grad_pre_l2.shape))
    print("Shape of  Gradb2:        {}".format(gradb2.shape))
    print("Shape of  GradW2:        {}".format(gradW2.shape))
    print("Shape of Grad_pre_l1:    {}".format(grad_pre_l1.shape))
    print("Shape of Gradb1:         {}".format(gradb1.shape))
    print("Shape of GradW1:         {}".format(gradW1.shape))
    print("Cross Entropy Error: {}".format(cost))
    print("\n")


    
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
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    print("No personal sanity checks implemented yet...")
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
