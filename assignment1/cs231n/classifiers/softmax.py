from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score_matrix = np.dot(X,W)
    exp_sm = np.exp(score_matrix)
    correct_exp_sm = exp_sm[np.arange(X.shape[0]), y]
    exp_sum_sm = np.sum(exp_sm, axis=1)
    L = -np.log(correct_exp_sm / exp_sum_sm)
    loss = np.mean(L)
    for i in range(X.shape[0]):
        # L_i = exp_sm[i][y[i]] / exp_sum_sm[i]
        for ci in range(W.shape[1]):
          dP_ci = 0
          dW_yi = 0
          dW_ci = 0
          if ci != y[i]:
            # dP_ci = -1 / (exp_sum_sm[i]**2) * exp_sm[i][ci] * X[i,:]
            # dW_ci = -1 / (correct_exp_sm / exp_sum_sm)[i] * dP_ci
            dW_ci = 1 / (exp_sum_sm[i] ) *exp_sm[i][ci] * X[i,:]
            dW[:,ci] += dW_ci
          if ci == y[i]:
             de_syi = correct_exp_sm[i] * X[i,:]
             dW_yi = -1/(exp_sum_sm[i] * correct_exp_sm[i]) * (1*exp_sum_sm[i]-correct_exp_sm[i]*1) * de_syi
             dW[:,y[i]] += dW_yi

    dW /= X.shape[0]
    dW += 2*reg*W




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score_matrix = np.dot(X,W)
    # stablization
    max_score = np.max(score_matrix, axis=1,keepdims=True)
    score_matrix -= max_score

    exp_sm = np.exp(score_matrix)
    correct_exp_sm = exp_sm[np.arange(X.shape[0]), y]
    exp_sum_sm = np.sum(exp_sm, axis=1)
    L = -np.log(correct_exp_sm / exp_sum_sm)
    loss = np.mean(L)

    temp = exp_sm / exp_sum_sm.reshape((X.shape[0],1))
    temp[np.arange(X.shape[0]), y] = (-1 + temp[np.arange(X.shape[0]),y])

    dW = np.dot(X.T, temp)

    dW /= X.shape[0]
    dW += 2*reg*W





    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
