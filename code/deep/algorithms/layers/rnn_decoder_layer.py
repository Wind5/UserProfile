import theano
import theano.tensor as tensor

from deep.algorithms.layers.layer import layer
from deep.algorithms.nodes.gru_node import GRUNode
from deep.algorithms.util import numpy_floatX

import theano.printing as printing



class DecoderLayer_Seq2Seq(layer):

    def __init__(self, word_embedding_dim, hidden_status_dim, tparams,
                 node_type=GRUNode, prefix='Decoder'):
        """
        Init the Decoder parameter: init_params.
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
        (self.sentence, self.mask, self.forward_hidden_status) = inputs

        assert self.mask is not None
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1

        last_h = tensor.alloc(self.forward_hidden_status[-1,:,:], n_samples, self.hidden_status_dim)
        state_below = tensor.dot(self.sentence, self.node.get_params_W())

        results, _ = theano.scan(self.node.node_update,
                                 sequences=[self.mask, state_below],
                                 outputs_info=[last_h],
                                 name=self._p(self.prefix, '_scan'),
                                 n_steps=n_steps)
        hidden_status_outputs = results
        
        return hidden_status_outputs

    

class DecoderLayer_Cho(layer):

    def __init__(self, word_embedding_dim, hidden_status_dim, tparams,
                 node_type=GRUNode, prefix='Decoder'):
        """
        Init the Decoder parameter: init_params.
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
        (self.sentence, self.mask, self.forward_hidden_status) = inputs

        assert self.mask is not None
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1

        last_h = tensor.alloc(numpy_floatX(0.), n_samples, self.hidden_status_dim)
        state_below = tensor.dot(tensor.concatenate([self.sentence, 
                                                     tensor.alloc(self.forward_hidden_status[-1,:,:], \
                                                                  self.sentence.shape[0], \
                                                                  self.forward_hidden_status.shape[1], \
                                                                  self.forward_hidden_status.shape[2])], 
                                                    axis=2),
                                 self.node.get_params_W())

        results, _ = theano.scan(self.node.node_update,
                                 sequences=[self.mask, state_below],
                                 outputs_info=[last_h],
                                 name=self._p(self.prefix, '_scan'),
                                 n_steps=n_steps)
        hidden_status_outputs = results
        
        return hidden_status_outputs
    
    
    
class DecoderMultiLayer_Seq2Seq(layer):

    def __init__(self, word_embedding_dim, hidden_status_dim, n_layers, tparams,
                 node_type=GRUNode, prefix='Decoder'):
        """
        Init the Decoder parameter: init_params.
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
        (self.sentence, self.mask, self.forward_hidden_status) = inputs

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
            last_h = self.forward_hidden_status[idx][-1]
            results, _ = theano.scan(self.node_list[idx].node_update,
                                     sequences=[self.mask, state_below],
                                     outputs_info=[last_h],
                                     name=self._p(self.prefix, '_scan'+str(idx)),
                                     n_steps=n_steps)
            hidden_states_list.append(results)
        hidden_status_outputs = hidden_states_list[-1]
        
        return hidden_status_outputs
    
    
    
class DecoderMultiLayer_Cho(layer):

    def __init__(self, word_embedding_dim, hidden_status_dim, n_layers, tparams,
                 node_type=GRUNode, prefix='Decoder'):
        """
        Init the Decoder parameter: init_params.
        """
        self.hidden_status_dim = hidden_status_dim
        self.word_embedding_dim = word_embedding_dim
        self.n_layers = n_layers
        self.prefix = prefix
        self.node_list = list()
        for idx in range(self.n_layers) :
            self.node_list.append(node_type( \
                word_embedding_dim=self.word_embedding_dim + idx*self.hidden_status_dim,
                hidden_status_dim=self.hidden_status_dim,
                tparams=tparams, prefix=self.prefix + str(idx)))

    
    def get_output(self, inputs):
        """
        Get outputs of encoder layer.
        Return all of the hidden status.
        """
        (self.sentence, self.mask, self.forward_hidden_status) = inputs

        assert self.mask is not None
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1
        
        hidden_states_list = list()
        for idx in range(self.n_layers) :
            sentence = tensor.concatenate([self.sentence, \
                                           tensor.alloc(self.forward_hidden_status[-1,:,:], \
                                                        self.sentence.shape[0], \
                                                        self.forward_hidden_status.shape[1], \
                                                        self.forward_hidden_status.shape[2])], \
                                          axis=2)
            for hidden_state in hidden_states_list :
                sentence = tensor.concatenate([sentence, hidden_state], axis=2)
            state_below = tensor.dot(sentence, self.node_list[idx].get_params_W())
            last_h = tensor.alloc(numpy_floatX(0.), n_samples, self.hidden_status_dim)
            results, _ = theano.scan(self.node_list[idx].node_update,
                                     sequences=[self.mask, state_below],
                                     outputs_info=[last_h],
                                     name=self._p(self.prefix, '_scan'+str(idx)),
                                     n_steps=n_steps)
            hidden_states_list.append(results)
        hidden_status_outputs = hidden_states_list[self.n_layers-1]
        
        return hidden_status_outputs
    
    
    
class LanguageModelLayer_Cho(layer):

    def __init__(self, word_embedding_dim, hidden_status_dim, tparams,
                 node_type=GRUNode, prefix='Decoder'):
        """
        Init the Decoder parameter: init_params.
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



class DecoderLayer_ChoT(layer):

    def __init__(self, word_embedding_dim, hidden_status_dim, layer_number, tparams,
                 node_type=GRUNode, prefix='Decoder'):
        """
        Init the Decoder parameter: init_params.
        """
        self.hidden_status_dim = hidden_status_dim
        self.prefix = prefix
        self.layer_number = layer_number
        self.node = node_type(word_embedding_dim=word_embedding_dim,
                              hidden_status_dim=hidden_status_dim, layer_number=self.layer_number,
                              tparams=tparams, prefix=self.prefix)

    
    def get_output(self, inputs):
        """
        Get outputs of encoder layer.
        Return all of the hidden status.
        """
        
        if len(inputs) == 4:
            (self.sentence, self.mask, self.forward_hidden_status, self.direction) = inputs
        
        if len (inputs) == 5:
            (self.sentence, self.mask, self.forward_hidden_status, self.direction, _) = inputs
                                    
        assert self.mask is not None
        
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1
        
        if len(inputs) == 4: 
            last_h = tensor.alloc(numpy_floatX(0.), n_samples, self.layer_number, self.hidden_status_dim)
           
        if len(inputs) == 5:    
            last_h = inputs[4]   
            last_h = tensor.alloc(last_h,
                                self.layer_number,
                                last_h.shape[0],
                                last_h.shape[1]).dimshuffle([1,0,2])

        state_below = tensor.dot(tensor.concatenate([self.sentence,
                                                     tensor.alloc(self.forward_hidden_status[-1, :, :],
                                                                  self.sentence.shape[0],
                                                                  self.forward_hidden_status.shape[1],
                                                                  self.hidden_status_dim)],
                                                    axis=2),
                                 self.node.get_params_W())

        
        
        results, _ = theano.scan(self.node.node_update,
                                 sequences=[self.mask, state_below, self.direction],
                                 outputs_info=[last_h],
                                 name=self._p(self.prefix, '_scan'),
                                 n_steps=n_steps)
        hidden_status_outputs = results
        
#         p = printing.Print('hidden_status_outputs')
#         hidden_status_outputs = p(hidden_status_outputs)        
        
        
        return hidden_status_outputs    
