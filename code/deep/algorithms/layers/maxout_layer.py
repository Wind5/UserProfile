import theano
import theano.tensor as T
import theano.tensor.signal.pool as downsample
import theano.printing as printing

from layer import layer
import deep.algorithms.util as util


class MaxoutLayer(layer):
    '''
    This class maxout each row of a matrix.
    '''
    def __init__(self, base_dim, refer_dim, tparams, prefix="maxout"):
        self.W_t = theano.shared(
            value=util.uniform_random_weight((refer_dim, 2 * refer_dim), 0.01),
            name=self._p(prefix, 'W_t'),
            borrow=True
        )
        self.W_o = theano.shared(
            value=util.uniform_random_weight((base_dim, refer_dim), 0.01),
            name=self._p(prefix, 'W_o'),
            borrow=True
        )
        # parameters of the model
        self.params = [self.W_t, self.W_o]
        if not tparams is None:
            tparams[self._p(prefix, 'W_t')] = self.W_t
            tparams[self._p(prefix, 'W_o')] = self.W_o
        else:
            print " tparams is None"
            
            
    def probability(self, base_data, refer_data):
        t_wave = T.dot(refer_data, self.W_t)
        t = downsample.pool_2d(t_wave, ds=(1, 2), ignore_border=True, mode='max')
        o = T.dot(base_data, self.W_o)
        p_y_given_x = T.dot(t, T.transpose(o))
        
        # This step is VERY IMPORTANT for the numerical stability.
        # Research usually focuses on points with idealized assumptions, however, a good project should consider all the
        # details which do not satisfy the assumptions, avoid it via elegant techniques. It should be based on but more robust
        # than research.
        
        # Thanks to 'http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/' and
        # <<Numerical Computing with IEEE Floating Point Arithmetic: Including One Theorem, One Rule of Thumb, and One Hundred and One Exercises>>
        row_max_p_y_given_x = p_y_given_x.max(axis=1).dimshuffle(0, 'x')
        
        p_y_given_x = p_y_given_x - row_max_p_y_given_x
        prob_matrix = T.nnet.softmax(p_y_given_x)
        
        return prob_matrix
            
            
    def likelihood(self, base_data, refer_data, y):
        prob_matrix = self.probability(base_data, refer_data)
        likelihood_vector = prob_matrix[T.arange(y.shape[0]), y]
        return likelihood_vector
            
            
    def get_output(self, base_data, refer_data):
        prob_matrix = self.probability(base_data, refer_data)
        y_pred_vector = T.argmax(prob_matrix, axis=1)
        return y_pred_vector, prob_matrix
