import numpy
import theano
import theano.tensor as tensor

from deep.algorithms.nodes.node import Node
from deep.algorithms.util import ortho_weight, numpy_floatX, uniform_random_weight



class RNNNode(Node):

    def __init__(self, word_embedding_dim, hidden_status_dim, tparams=None, prefix='RNN'):
        """
        Init the RNN parameter: init_params.
        Updation in GRU :
            step1. h(t) = f(W_r dot x(t) + U_r dot h(t-1)).
        We can combine W and C into one tensor W
        """
        self.hidden_status_dim = hidden_status_dim
        self.params = tparams
        self.prefix = prefix
        W_bound = 0.01

        # combine step1~3 W dot t, so W's dimension is (word_embedding_dim, hidden_status_dim)
        W = uniform_random_weight(size=(word_embedding_dim, hidden_status_dim), bound=W_bound)
        # combine step1~2 U dot h, so U's dimension is (hidden_status_dim, 2)
        # connot combine step1~3, so split U_rh
        U = numpy.concatenate([ortho_weight(hidden_status_dim)], axis=1)
        
        if tparams is not None: 
            tparams[self._p(prefix, 'W')] = theano.shared(W, name=self._p(prefix, 'W'))
            tparams[self._p(prefix, 'U')] = theano.shared(U, name=self._p(prefix, 'U'))
        else:
            print 'tparams is None'
    

    def node_update(self, m_, x_, h_) :
        """
        Update params in RNN.
        """
        preact = tensor.dot(h_, self.params[self._p(self.prefix, 'U')])
        preact += x_
        h = tensor.tanh(preact)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
    
        return h

    
    def get_params_W(self) :
        return self.params[self._p(self.prefix, 'W')]
