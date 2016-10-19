import theano
import theano.tensor as tensor

from deep.algorithms.layers.layer import layer
from deep.algorithms.util import numpy_floatX, uniform_random_weight



class MemoryLayer(layer):

    def __init__(self, n_words, word_embedding_dim, represent, isfirst, tparams, prefix):
        """
        Init the Memory parameter: init_params.
        """
        self.word_embedding_dim = word_embedding_dim
        self.params = tparams
        self.prefix = prefix
        EmbA = uniform_random_weight(size=(n_words, word_embedding_dim), bound=0.01)
        EmbB = uniform_random_weight(size=(n_words, word_embedding_dim), bound=0.01)
        EmbC = uniform_random_weight(size=(n_words, word_embedding_dim), bound=0.01)
        TemA = uniform_random_weight(size=(500, word_embedding_dim), bound=0.01)
        TemC = uniform_random_weight(size=(500, word_embedding_dim), bound=0.01)
        
        if tparams is not None: 
            tparams[self._p(prefix, 'EmbA')] = theano.shared(EmbA, name=self._p(prefix, 'EmbA'))
            if isfirst == True :
                tparams[self._p(prefix, 'EmbB')] = theano.shared(EmbB, name=self._p(prefix, 'EmbB'))
            tparams[self._p(prefix, 'EmbC')] = theano.shared(EmbC, name=self._p(prefix, 'EmbC'))
            if represent == 'PE' :
                tparams[self._p(prefix, 'TemA')] = theano.shared(TemA, name=self._p(prefix, 'TemA'))
                tparams[self._p(prefix, 'TemC')] = theano.shared(TemC, name=self._p(prefix, 'TemC'))

    
    def get_output(self, inputs, represent='PE', linear=True):
        """
        Get outputs of memory layer.
        Return all of the internal state.
        """
        (conditions, conditions_mask, temporal, temporal_mask, query, query_mask, u) = inputs
        [n_samples, n_conds, n_condition_times] = conditions.shape
        n_query_times = query.shape[1]
        conditions_mask = tensor.stack([conditions_mask]*self.word_embedding_dim, axis=3)
        query_mask = tensor.stack([query_mask]*self.word_embedding_dim, axis=2)
        temporal_mask = tensor.stack([temporal_mask]*self.word_embedding_dim, axis=2)
        
        # conditions(n_samples, n_conds, n_condition_times) -> m(n_samples, n_conds, word_embedding_dim)
        self.m = self.params[self._p(self.prefix, 'EmbA')][conditions.flatten()].reshape( \
            [n_samples, n_conds, n_condition_times, self.word_embedding_dim])
        self.m = self.m * conditions_mask
        self.m = tensor.sum(self.m, axis=2)
        # m(n_samples, n_conds, word_embedding_dim) + PE(n_samples, n_conds, word_embedding_dim)
        if represent == 'PE' :
            peA = self.params[self._p(self.prefix, 'TemA')][temporal.flatten()].reshape( \
                [n_samples, n_conds, self.word_embedding_dim])
            peA = peA * temporal_mask
            self.m = self.m + peA
        if u is None :
            # query(n_samples, n_query_times) -> u(n_samples, word_embedding_dim)
            self.u = self.params[self._p(self.prefix, 'EmbB')][query.flatten()].reshape( \
                [n_samples, n_query_times, self.word_embedding_dim])
            self.u = self.u * query_mask
            self.u = tensor.sum(self.u, axis=1)
        else :
            self.u = u
        # softmax(u.m) -> p(n_samples, n_conds)
        self.p = self.m * self.u.dimshuffle(0,'x',1)
        self.p = tensor.sum(self.p, axis=2)
        if linear != True :
            self.p = tensor.nnet.nnet.softmax(self.p)
        # conditions(n_samples, n_conds, n_condition_times) -> c(n_samples, n_conds, word_embedding_dim)
        self.c = self.params[self._p(self.prefix, 'EmbC')][conditions.flatten()].reshape( \
            [n_samples, n_conds, n_condition_times, self.word_embedding_dim])
        self.c = self.c * conditions_mask
        self.c = tensor.sum(self.c, axis=2)
        # c(n_samples, n_conds, word_embedding_dim) + PE(n_samples, n_conds, word_embedding_dim)
        if represent == 'PE' :
            peC = self.params[self._p(self.prefix, 'TemC')][temporal.flatten()].reshape( \
                [n_samples, n_conds, self.word_embedding_dim])
            peC = peC * temporal_mask
            self.c = self.c + peC
        # o(n_samples, word_embedding_dim) + u(n_samples, word_embedding_dim)
        self.o = self.c * self.p.dimshuffle(0,1,'x')
        self.o = tensor.sum(self.o, axis=1)
        self.o = self.o + self.u
        
        return self.o