import numpy
import theano
import theano.tensor as tensor

from deep.algorithms.nodes.node import Node
from deep.algorithms.util import ortho_weight, numpy_floatX, uniform_random_weight
import theano.printing as printing


class GRUNodeRoot2Leaf(Node):

    def __init__(self, word_embedding_dim, hidden_status_dim, layer_number, tparams=None, prefix='GRU'):
        """
        Init the GRU parameter: init_params.
        Updation in GRU :
            step1. r(t) = f(W_r dot x(t) + U_r dot h(t-1) + C_r dot h_last).
            step2. z(t) = f(W_z dot x(t) + U_z dot h(t-1) + C_z dot h_last).
            step3. h_wave(t) = f(W dot x(t) + U dot (r(t) * h(t-1)) + C dot h_last).
            step4. h(t) = (1-z(t)) * h(t-1) + z(t) * h_wave(t).
        We can combine W and C into one tensor W
        """
        self.hidden_status_dim = hidden_status_dim
        self.params = tparams
        self.prefix = prefix
        self.layer_number = layer_number
        W_bound = 0.01

        # combine step1~3 W dot t, so W's dimension is (word_embedding_dim, hidden_status_dim, 3)
        W = uniform_random_weight(size=(layer_number, word_embedding_dim, 3 * hidden_status_dim), bound=W_bound)
        # combine step1~2 U dot h, so U's dimension is (hidden_status_dim, 2)
        # connot combine step1~3, so split U_rh
        
        U = uniform_random_weight(size=(layer_number, hidden_status_dim, hidden_status_dim * 2), bound=W_bound)
        U_rh = uniform_random_weight(size=(layer_number, hidden_status_dim, hidden_status_dim), bound=W_bound)
        
        if tparams is not None: 
            tparams[self._p(prefix, 'W')] = theano.shared(W, name=self._p(prefix, 'W'))
            tparams[self._p(prefix, 'U')] = theano.shared(U, name=self._p(prefix, 'U'))
            tparams[self._p(prefix, 'U_rh')] = theano.shared(U_rh, name=self._p(prefix, 'U_rh'))
        else:
            print ' tparams is None'
    

    def node_update(self, m_, x_, d_, h_) :
        """
        Update params in GRU.
        """
        def _slice(_x, n, dim):
            return _x[:, :, n * dim:(n + 1) * dim]
        
        h_current_direction = h_[tensor.arange(h_.shape[0]), d_]
        
        preact = tensor.dot(h_current_direction, self.params[self._p(self.prefix, 'U')])
        preact += _slice(x_, 0 , 2 * self.hidden_status_dim)
        z = tensor.nnet.sigmoid(_slice(preact, 0, self.hidden_status_dim))
        r = tensor.nnet.sigmoid(_slice(preact, 1, self.hidden_status_dim))
        
        rh_temp_list = list()
        for i in range(self.layer_number):
            U_rh = self.params[self._p(self.prefix, 'U_rh')]
            
            temp = (r[:, i, :] * h_current_direction) 
            temp = tensor.dot(temp, U_rh[i, :, :])
            rh_temp_list.append(temp.dimshuffle([0, 'x', 1]))
            
        h_wave = tensor.concatenate(rh_temp_list, axis=1) + _slice(x_, 2 , self.hidden_status_dim)
        h_wave = tensor.tanh(h_wave)
        h = (1 - z) * h_ + z * h_wave
        h = m_[:, None, None] * h + (1. - m_)[:, None, None] * h_
    
        return h

    
    def get_params_W(self) :
        return self.params[self._p(self.prefix, 'W')]
