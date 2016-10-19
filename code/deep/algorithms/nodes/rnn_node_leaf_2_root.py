import numpy
import theano
import theano.tensor as tensor

from deep.algorithms.nodes.node import Node
from deep.algorithms.util import ortho_weight, numpy_floatX, uniform_random_weight
import theano.printing as printing


class GRUNodeLeaf2Root(Node):

    def __init__(self, word_embedding_dim, hidden_status_dim, tparams=None, prefix='GRUTree'):
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
        self.triple_hidden_status_dim = 3 * hidden_status_dim
        
        
        self.params = tparams
        self.prefix = prefix
        self.word_embedding_dim = word_embedding_dim
        W_bound = 0.01

        # combine step1~3 W dot t, so W's dimension is (word_embedding_dim, hidden_status_dim, 3)
        W = uniform_random_weight(size=(word_embedding_dim, self.hidden_status_dim), bound=W_bound)
        # combine step1~2 U dot h, so U's dimension is (hidden_status_dim, 2)
        # connot combine step1~3, so split U_rh
        
        U = uniform_random_weight(size=(self.triple_hidden_status_dim, self.hidden_status_dim), bound=W_bound)
        
        if tparams is not None: 
            tparams[self._p(prefix, 'W')] = theano.shared(W, name=self._p(prefix, 'W'))
            tparams[self._p(prefix, 'U')] = theano.shared(U, name=self._p(prefix, 'U'))
        else:
            print ' tparams is None'
    

    def get_non_seq_parameter(self, n_samples):
        index_seq = tensor.arange(n_samples)
        eob_matrix = tensor.alloc(0.,
                                  n_samples, 3, self.hidden_status_dim)
        return [index_seq, eob_matrix]

    def node_update(self, m_, x_, children, children_mask, h_buffer, last_h,
                     index_seq, eob_matrix) :
        """
        Update params in GRU.
        """
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]
        h_0 = h_buffer[index_seq.dimshuffle([0, 'x']), children] * children_mask + (1. - children_mask) * eob_matrix
        h_ = h_0.reshape([h_0.shape[0], 3 * self.hidden_status_dim])        
        U = self.params[self._p(self.prefix, 'U')]
        h_wave = tensor.tanh(tensor.dot(h_, U) + x_)
        h = m_[:, None] * h_wave + (1. - m_)[:, None] * last_h
        h_buffer = tensor.concatenate([h_buffer[:, 1:], h.dimshuffle([ 0, 'x', 1])], axis=1)
        return h_buffer, h

    
    def get_params_W(self) :
        return self.params[self._p(self.prefix, 'W')]
