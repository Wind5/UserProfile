﻿"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T

from layer import layer

import deep.util.config as config

from theano import printing


class SoftmaxLayer(layer):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out, tparams, prefix="softmax"):
        """ Initialize the parameters of the logistic regression

        :type input_data: theano.tensor.TensorType
        :param input_data: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=config.globalFloatType()
            ),
            name=self._p(prefix, 'W'),
            borrow=True
        )
        
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=config.globalFloatType()
            ),
            name=self._p(prefix, 'b'),
            borrow=True
        )
        
        # parameters of the model
        self.params = tparams
        self.prefix = prefix
        if not tparams is None:
            tparams[self._p(prefix, 'W')] = self.W
            tparams[self._p(prefix, 'b')] = self.b
        else:
            print " tparams is None"

    def probability(self, input_data):
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        
        # For numerical stability.
        p0 = T.dot(input_data, self.W) + self.b
        row_max_p0 = p0.max(axis=1).dimshuffle(0, 'x')
        prob_matrix = T.nnet.softmax(p0 - row_max_p0)

        return prob_matrix

    def likelihood(self, input_data, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        
        prob_matrix = self.probability(input_data)
        likelihood_vector = prob_matrix[T.arange(y.shape[0]), y]
        
        return likelihood_vector
    

    def cross_entropy(self, input_data, y):
        """Return the cross entropy of input_data and y.
        """
        prob_matrix = \
            T.nnet.softmax(T.dot(input_data, self.params[self._p(self.prefix, 'W')]) + \
                           self.params[self._p(self.prefix, 'b')])
        cross_entropy = T.nnet.categorical_crossentropy(prob_matrix, y) 
        
        return cross_entropy


    def get_output(self, input_data):
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        prob_matrix = \
            T.nnet.softmax(T.dot(input_data, self.params[self._p(self.prefix, 'W')]) + \
                           self.params[self._p(self.prefix, 'b')])

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        y_pred_vector = T.argmax(prob_matrix, axis=1)
        # end-snippet-1
        return y_pred_vector, prob_matrix
