__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T

from layer import layer

import deep.util.config as config

from theano import printing


class SingleHiddenLayer(layer):

    def __init__(self, n_in, n_out, tparams, prefix="SingleHiddenLayer"):
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
        self.params = [self.W, self.b]
        if not tparams is None:
            tparams[self._p(prefix, 'W')] = self.W
            tparams[self._p(prefix, 'b')] = self.b
        else:
            print " tparams is None"


    def get_output(self, input_data):
        p0 = T.dot(input_data, self.W) + self.b
        alpha_vector = p0.exp().log1p()

        return alpha_vector
