import numpy
import theano
import theano.tensor as tensor

from deep.algorithms.nodes.node import Node
from deep.algorithms.util import ortho_weight, numpy_floatX, uniform_random_weight
import theano.printing as printing


class GRUNodeLeaf2Root(Node):

    def __init__(self, word_embedding_dim, hidden_status_dim, tparams=None, prefix='GRU'):
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
        self.word_embedding_dim = word_embedding_dim
        W_bound = 0.1

        # combine step1~3 W dot t, so W's dimension is (word_embedding_dim, hidden_status_dim, 3)
        W = uniform_random_weight(size=(word_embedding_dim, 3 * hidden_status_dim), bound=W_bound)
        # combine step1~2 U dot h, so U's dimension is (hidden_status_dim, 2)
        # connot combine step1~3, so split U_rh
        
        U = uniform_random_weight(size=(hidden_status_dim, 2 * hidden_status_dim), bound=W_bound)
        U_rh = uniform_random_weight(size=(hidden_status_dim, hidden_status_dim), bound=W_bound)
        U_union = uniform_random_weight(size=(3 * hidden_status_dim, hidden_status_dim), bound=W_bound)
        
        
        if tparams is not None: 
            tparams[self._p(prefix, 'W')] = theano.shared(W, name=self._p(prefix, 'W'))
            tparams[self._p(prefix, 'U')] = theano.shared(U, name=self._p(prefix, 'U'))
            tparams[self._p(prefix, 'U_rh')] = theano.shared(U_rh, name=self._p(prefix, 'U_rh'))
            tparams[self._p(prefix, 'U_union')] = theano.shared(U_union, name=self._p(prefix, 'U_union'))
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

        h_0 = h_buffer[index_seq.dimshuffle([0, 'x']), children] * children_mask \
                                                + (1. - children_mask) * eob_matrix
        h_ = h_0.reshape([h_0.shape[0], 3 * self.hidden_status_dim])  
        U_union = self.params[self._p(self.prefix, 'U_union')]
        h_ = tensor.tanh(tensor.dot(h_, U_union))
        
        preact = tensor.dot(h_, self.params[self._p(self.prefix, 'U')])
        preact += _slice(x_, 0 , 2 * self.hidden_status_dim)
         
        z = tensor.nnet.sigmoid(_slice(preact, 0, self.hidden_status_dim))
        r = tensor.nnet.sigmoid(_slice(preact, 1, self.hidden_status_dim))
         
        h_wave = theano.dot(r * h_, \
                            self.params[self._p(self.prefix, 'U_rh')]) \
                            + _slice(x_, 2, self.hidden_status_dim)
        h_wave = tensor.tanh(h_wave)
        
        h = (1 - z) * h_ + z * h_wave        
        h = m_[:, None] * h + (1. - m_)[:, None] * last_h
        h_buffer = tensor.concatenate([h_buffer[:, 1:], h.dimshuffle([ 0, 'x', 1])], axis=1)
        return h_buffer, h

    
    def get_params_W(self) :
        return self.params[self._p(self.prefix, 'W')]
