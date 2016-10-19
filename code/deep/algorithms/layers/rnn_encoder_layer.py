import theano
import theano.tensor as tensor

from deep.algorithms.layers.layer import layer
from deep.algorithms.nodes.gru_node import GRUNode
from deep.algorithms.nodes.gru_node_leaf_2_root import GRUNodeLeaf2Root
from deep.algorithms.util import numpy_floatX



class EncoderLayer(layer):

    def __init__(self, word_embedding_dim, hidden_status_dim, tparams, 
                 node_type=GRUNode , prefix='Encoder'):
        """
        Init the Encoder parameter: init_params.
        """
        self.hidden_status_dim = hidden_status_dim
        self.prefix = prefix
        self.node = node_type(word_embedding_dim=word_embedding_dim, 
                              hidden_status_dim=hidden_status_dim, 
                              tparams=tparams, prefix=self.prefix)

    
    def get_output(self, inputs):
        """
        Get outputs of encoder layer.
        Return all of the hidden status.
        """
        (self.sentence, self.mask) = inputs
        
        assert self.mask is not None
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1

        last_h = tensor.alloc(numpy_floatX(0.), n_samples, self.hidden_status_dim)
        state_below = tensor.dot(self.sentence, self.node.get_params_W())

        results, _ = theano.scan(self.node.node_update,
                                 sequences=[self.mask, state_below],
                                 outputs_info=[last_h],
                                 name=self._p(self.prefix, '_scan'),
                                 n_steps=n_steps)
        hidden_status_outputs = results
        
        return hidden_status_outputs
    
    
    
class EncoderMultiLayer(layer):

    def __init__(self, word_embedding_dim, hidden_status_dim, n_layers, tparams, 
                 node_type=GRUNode , prefix='Encoder'):
        """
        Init the Encoder parameter: init_params.
        """
        self.hidden_status_dim = hidden_status_dim
        self.word_embedding_dim = word_embedding_dim
        self.n_layers = n_layers
        self.prefix = prefix
        self.node_list = [node_type(word_embedding_dim=self.word_embedding_dim, \
                                    hidden_status_dim=self.hidden_status_dim, \
                                    tparams=tparams, prefix=self.prefix + str(0))]
        for idx in range(1, self.n_layers) :
            self.node_list.append(node_type(word_embedding_dim=self.hidden_status_dim, \
                                            hidden_status_dim=self.hidden_status_dim, \
                                            tparams=tparams, prefix=self.prefix + str(idx)))

    
    def get_output(self, inputs):
        """
        Get outputs of encoder layer.
        Return all of the hidden status.
        """
        (self.sentence, self.mask) = inputs
        
        assert self.mask is not None
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1

        hidden_states_list = [self.sentence]
        for idx in range(self.n_layers) :
            sentence = hidden_states_list[idx]
            state_below = tensor.dot(sentence, self.node_list[idx].get_params_W())
            last_h = tensor.alloc(numpy_floatX(0.), n_samples, self.hidden_status_dim)
            results, _ = theano.scan(self.node_list[idx].node_update,
                                     sequences=[self.mask, state_below],
                                     outputs_info=[last_h],
                                     name=self._p(self.prefix, '_scan'+str(idx)),
                                     n_steps=n_steps)
            hidden_states_list.append(results)
        hidden_status_outputs = hidden_states_list
        
        return hidden_status_outputs
    
    
    
class TreeEncoderLayer(layer):
    def __init__(self, word_embedding_dim, hidden_status_dim, tparams, prefix='Encoder'):
        """
        Init the Encoder parameter: init_params.
        """
        self.hidden_status_dim = hidden_status_dim
        self.prefix = prefix
        self.node = GRUNodeLeaf2Root(word_embedding_dim=word_embedding_dim,
                              hidden_status_dim=hidden_status_dim,
                              tparams=tparams, prefix=self.prefix)

    
    def get_output(self, inputs):
        """
        Get outputs of encoder layer.
        Return all of the hidden status.
        """
        (self.sentence, self.mask, self.question_children, self.question_children_mask, self.max_offset) = inputs
        
        assert self.mask is not None
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1

        queue_buffer = tensor.alloc(numpy_floatX(0.), n_samples,
                            self.max_offset, self.hidden_status_dim)
        state_below = tensor.dot(self.sentence, self.node.get_params_W())

        non_seq = self.node.get_non_seq_parameter(n_samples)
        self.question_children_mask= self.question_children_mask.dimshuffle([0, 1, 2, 'x'])
        results, _ = theano.scan(self.node.node_update,
             sequences=[self.mask, state_below, self.question_children, self.question_children_mask],
             outputs_info=[queue_buffer, queue_buffer[:, -1, :]],
             non_sequences=non_seq,
             name=self._p(self.prefix, '_scan'),
             n_steps=n_steps)
        hidden_status_outputs = results[1]
        return hidden_status_outputs
