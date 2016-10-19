import numpy
import theano
import theano.tensor as tensor
import theano.tensor.signal.pool as downsample
import theano.tensor.signal.conv as conv

from deep.algorithms.layers.layer import layer
from deep.algorithms.nodes.gru_node import GRUNode
from deep.algorithms.util import numpy_floatX



class EncoderLayer(layer):

    def __init__(self, rng, filter_shape, image_shape, pool_size, tparams, prefix='Encoder'):
        """
        Init the Encoder parameter: init_params.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.pool_shape = ()
        self.prefix = prefix
        self.tparams = tparams
        
        fan_in = numpy.prod(self.filter_shape[1:])
        fan_out = self.filter_shape[0] * numpy.prod(self.filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape),
                               dtype=theano.config.floatX)
        b_values = numpy.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.b = b_values
        # store parameters of this layer
        if tparams is not None: 
            tparams[self._p(self.prefix, 'W')] = theano.shared(self.W, name=self._p(self.prefix, 'W'))
            tparams[self._p(self.prefix, 'b')] = theano.shared(self.b, name=self._p(self.prefix, 'b'))
        else:
            print ' tparams is None'

    
    def get_output(self, inputs):
        """
        Get outputs of encoder layer.
        Return all of the hidden status.
        """
        (self.sentence, self.mask) = inputs
        
        conv_out = conv.conv2d(input=self.sentence, \
                               filters=self.tparams[self._p(self.prefix, 'W')], \
                               filter_shape=self.filter_shape, \
                               image_shape=self.image_shape)
        conv_out = conv_out * self.mask.dimshuffle(0,'x',1,'x')
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.pool_2d(input=conv_out, ds=self.pool_size, \
                                        ignore_border=True, mode='max')
        output = tensor.tanh(pooled_out + self.tparams[self._p(self.prefix, 'b')]\
                             .dimshuffle('x', 0, 'x', 'x'))
        
        return output[:,:,0,0], conv_out