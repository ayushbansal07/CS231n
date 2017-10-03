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
  num_ex = X.shape[0]
  gradY = np.zeros((num_ex,W.shape[1]))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  for i in range(0,num_ex):
    scores = X[i][np.newaxis].dot(W)
    #print(np.sum(np.exp(scores),axis=1))
    nr = np.exp(scores[0,y[i]])
    dr = np.sum(np.exp(scores))
    smax = nr / dr
    #print(smax)
    loss -= np.log(smax)
    gradY[i] = np.exp(scores)/dr
    gradY[i][y[i]] -= 1
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW = np.transpose(X).dot(gradY)
  dW/=num_ex
  dW += 2*reg*W
  loss/=num_ex
  loss += 2*reg* np.sum(W*W)

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_ex = X.shape[0]
  dW = np.zeros_like(W)
  gradY = np.zeros((num_ex,W.shape[1]))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  scores = X.dot(W)
  correctLabels = [x for x in y]
  loss-=sum(scores[np.arange(num_ex),y])
  epowerscores = np.exp(scores)
  sigmascores = np.sum(epowerscores,axis=1)
  loss+=sum(np.log(sigmascores))

  gradY = epowerscores/sigmascores[:,np.newaxis]
  gradY[np.arange(num_ex),y]-=1
  #print(gradY)
  dW = np.transpose(X).dot(gradY)
  dW/=num_ex
  dW += 2*reg*W

  loss/=num_ex
  loss+=2*reg*np.sum(W*W)

  #print(scores)
  #print(sum(scores[correctLabels]))
  #loss-=sum(scores[correctLabels])
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

