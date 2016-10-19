import numpy
import theano
import theano.tensor as tensor

from deep.algorithms.layers.layer import layer
from deep.algorithms.nodes.gru_node import GRUNode
from deep.algorithms.nodes.attention_node import AttentionNode
from deep.algorithms.util import ortho_weight, numpy_floatX, uniform_random_weight



class AttentionDecoderLayer(layer):

    def __init__(self, word_embedding_dim, hidden_status_dim, encoder_hidden_dim,
                 tparams, prefix='Decoder', node_type=GRUNode):
        """
        Init the Decoder parameter: init_params.
        """
        self.params = tparams
        W_bound = numpy.sqrt(6. / (hidden_status_dim))
        W = uniform_random_weight(size=(hidden_status_dim, hidden_status_dim), bound=W_bound)
        # if tparams is not None: 
        #     tparams[self._p(prefix, 'Ws')] = theano.shared(W, name=self._p(prefix, 'Ws'))
        self.hidden_status_dim = hidden_status_dim
        self.prefix = prefix
        self.node = node_type(word_embedding_dim=word_embedding_dim,
                              hidden_status_dim=hidden_status_dim,
                              tparams=tparams, prefix=self._p(self.prefix, 'GRU'))
        self.attention_node = AttentionNode(word_embedding_dim=word_embedding_dim,
                                            hidden_status_dim=hidden_status_dim,
                                            encoder_hidden_dim = encoder_hidden_dim,
                                            tparams=tparams, prefix=self._p(self.prefix, 'Attention'))


    def get_output(self, inputs):
        """
        Get outputs of encoder layer.
        Return all of the hidden status.
        """
        (self.sentence, self.mask, self.encoder_hidden_status, self.question_mask) = inputs
        
        assert self.mask is not None
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1
            
        # last_s = tensor.dot(self.encoder_hidden_status[0, :, self.hidden_status_dim:],
        #                     self.params[self._p(self.prefix, 'Ws')])
        last_s = tensor.alloc(numpy_floatX(0.), n_samples, self.hidden_status_dim)
        state_below = self.sentence

        def upd(am_, x_, s_, h_, qm_):
            c, alpha = self.attention_node.node_update(s_, h_, qm_)
            x_ = tensor.dot(tensor.concatenate([x_, c], axis=1), self.node.get_params_W())
            s = self.node.node_update(am_, x_, s_)
            
            return s, c, alpha

        results, _ = theano.scan(upd,
                                 sequences=[self.mask, state_below],
                                 outputs_info=[last_s, None, None],
                                 non_sequences=[self.encoder_hidden_status, self.question_mask],
                                 name=self._p(self.prefix, '_scan'),
                                 n_steps=n_steps)
        hidden_status_outputs, context_outputs, alpha_outputs = results
        
        return hidden_status_outputs, context_outputs, alpha_outputs
