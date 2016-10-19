# -*- encoding = utf-8 -*-
from collections import OrderedDict

import numpy
import theano
import theano.tensor as tensor
import theano.printing as printing
from scipy.stats.stats import pearsonr

import deep.util.config as config
from deep.algorithms.util import numpy_floatX
from deep.algorithms.networks.network import Network
from deep.algorithms.nodes.gru_node import GRUNode
# from algorithms.layers.softmax_layer import SoftmaxLayer
from deep.algorithms.layers.maxout_layer import MaxoutLayer
from deep.algorithms.layers.rnn_encoder_layer import EncoderLayer
from deep.algorithms.layers.rnn_decoder_layer import DecoderLayer_Cho



class ChoEncoderDecoderNetwork(Network):
    """
    This class will process the dialog pair with a encoder-decoder network.
    It has 2 abilities:
        1. Train the language model.
        2. Model the relationship of Q&A
    """

    def init_global_params(self, options):
        """
        Global (not LSTM) parameter. For the embeding and the classifier.
        """
        params = OrderedDict()
        randn = numpy.random.rand(options['n_words'], options['word_embedding_dim'])
        params['Wemb_e'] = (0.01 * randn).astype(config.globalFloatType()) 
        randn = numpy.random.rand(options['n_words'], options['word_embedding_dim'])
        params['Wemb_d'] = (0.01 * randn).astype(config.globalFloatType()) 

        return params


    def __init__(self, n_words, word_embedding_dim, hidden_status_dim, input_params):
        self.options = options = {
            'n_words': n_words,
            'word_embedding_dim': word_embedding_dim,
            'hidden_status_dim': hidden_status_dim,
            'learning_rate': 0.0001
            }
        # global paramters
        params = self.init_global_params(options)
        # Theano paramters
        self.tparams = self.init_tparams(params)

        # construct network
        self.question = tensor.matrix('question', dtype='int64')
        self.question_mask = tensor.matrix('question_mask', dtype=config.globalFloatType())
        self.question_embedding = self.tparams['Wemb_e'][self.question.flatten()].reshape(
            [self.question.shape[0], self.question.shape[1], options['word_embedding_dim']])
        self.answer = tensor.matrix('answer', dtype='int64')
        self.answer_mask = tensor.matrix('answer_mask', dtype=config.globalFloatType())
        self.answer_embedding = self.tparams['Wemb_d'][self.answer.flatten()].reshape(
            [self.answer.shape[0], self.answer.shape[1], options['word_embedding_dim']])
        '''
        theano.config.compute_test_value = 'off'
        self.question.tag.test_value = numpy.array([[10, 2, 0], [5, 9, 2]]) # for debug
        self.question_mask.tag.test_value = numpy.array([[1, 1, 0], [1, 1, 1]]) # for debug
        self.answer.tag.test_value = numpy.array([[11, 10, 2], [5, 2, 0]]) # for debug
        self.answer_mask.tag.test_value = numpy.array([[1, 1, 1], [1, 1, 0]]) # for debug
        '''
        #   1. encoder layer
        self.encoder_layer = EncoderLayer(word_embedding_dim=options['word_embedding_dim'],
                                          hidden_status_dim=options['hidden_status_dim'],
                                          tparams=self.tparams, node_type=GRUNode)
        #   2. decoder layer
        self.decoder_layer = \
            DecoderLayer_Cho(word_embedding_dim=options['word_embedding_dim'] + \
                             options['hidden_status_dim'], \
                             hidden_status_dim=options['hidden_status_dim'],
                             tparams=self.tparams, node_type=GRUNode)
        #   3. maxout layer
        self.maxout_layer = MaxoutLayer(base_dim=options['word_embedding_dim'],
                                        refer_dim=2 * options['hidden_status_dim'] + \
                                         options['word_embedding_dim'], \
                                         tparams=self.tparams, prefix='maxout')
        
        #   1. encoder layer
        self.encoder_hidden_status = \
            self.encoder_layer.get_output(inputs=(self.question_embedding, self.question_mask))
        #   2. decoder layer
        self.decoder_hidden_status = \
            self.decoder_layer.get_output(inputs=(self.answer_embedding, self.answer_mask, \
                                                  self.encoder_hidden_status))
        #   3. maxout layer
        self.maxout_input = \
            tensor.concatenate([self.decoder_hidden_status[:-1, :, :]\
                                .reshape([(self.answer.shape[0] - 1) * self.answer.shape[1], \
                                          options['hidden_status_dim']]), \
                                tensor.alloc(self.encoder_hidden_status[-1, :, :], \
                                             self.answer.shape[0] - 1, \
                                             self.answer.shape[1], \
                                             options['hidden_status_dim'])\
                                .reshape([(self.answer.shape[0] - 1) * self.answer.shape[1], \
                                          options['hidden_status_dim']]), \
                                self.answer_embedding[:-1, :, :]\
                                .reshape([(self.answer.shape[0] - 1) * self.answer.shape[1], \
                                          options['word_embedding_dim']])], \
                               axis=1)
        likihood_vector = \
            self.maxout_layer.likelihood(base_data=self.tparams['Wemb_d'], \
                                         refer_data=self.maxout_input, \
                                         y=self.answer[1:, :].flatten())
        # get evaluation and cost
        likihood_vector = -tensor.log(likihood_vector)
        self.cost = tensor.dot(likihood_vector.flatten(), self.answer_mask[1:, :].flatten()) \
            / self.answer_mask[1:, :].sum()
        prob_matrix = likihood_vector.reshape([self.answer_mask[1:,:].shape[0], \
                                               self.answer_mask[1:,:].shape[1]])
        self.likihood_vector = tensor.sum(prob_matrix * self.answer_mask[1:, :], axis=0) \
            / tensor.sum(self.answer_mask[1:,:], axis=0)
        
        self.set_parameters(input_params)  # params from list to TensorVirable
    


    def get_training_function(self, cr, batch_size, batch_repeat=1):
        lr = tensor.scalar(name='lr')
        grads = tensor.grad(self.cost, wrt=self.tparams.values())
        f_grad_shared, f_update = self.adadelta(lr, self.tparams, grads, \
                                                [self.question, self.question_mask, \
                                                 self.answer, self.answer_mask], \
                                                [self.cost])
        
        def update_function(index):
            question, question_mask, answer, answer_mask = \
                cr.get_trainset([index * batch_size, (index + 1) * batch_size])
            for _ in xrange(batch_repeat):
                cost = f_grad_shared(question, question_mask, answer, answer_mask)
                f_update(self.options["learning_rate"])
            return cost
        
        return update_function
    
    

    def get_validing_function(self, cr, batch_size=200):
        valid_function = self.get_cost_function()
        
        def update_function():
            n_validset = cr.get_size()[1]
            n_batches = (n_validset - 1) / batch_size + 1
            cost = 0.0
            for index in range(n_batches) :
                question, question_mask, answer, answer_mask = \
                    cr.get_validset([index * batch_size, (index + 1) * batch_size])
                cost += valid_function(question, question_mask, answer, answer_mask)[0]
            cost = cost / n_batches
            return [cost]
        
        return update_function
    

    def get_testing_function(self, cr, batch_size=100):
        test_function = self.get_cost_function()
        
        def update_function():
            n_testset = cr.get_size()[2]
            n_batches = (n_testset - 1) / batch_size + 1
            cost = 0.0
            for index in range(n_batches) :
                question, question_mask, answer, answer_mask = \
                    cr.get_testset([index * batch_size, (index + 1) * batch_size])
                cost += test_function(question, question_mask, answer, answer_mask)[0]
            cost = cost / n_batches
            return [cost]
        return update_function
    

    def get_pr_function(self, cr, batch_size=100):
        pr_function = theano.function(inputs=[self.question, self.question_mask, \
                                              self.answer, self.answer_mask], \
                                      outputs=[self.likihood_vector], name='pr_function', \
                                      on_unused_input='ignore')
        
        def update_function():
            n_prset = cr.get_size()[2]
            n_batches = (n_prset - 1) / batch_size + 1
            cost_list = list()
            for index in range(n_batches) :
                question, question_mask, answer, answer_mask = \
                    cr.get_prset([index * batch_size, (index + 1) * batch_size])
                score_list = pr_function(question, question_mask, answer, answer_mask)[0]
                cost_list.extend(score_list)
            corre_score = pearsonr(cost_list, cr.get_pr_score()) 
            return [corre_score[0]]
        return update_function
    

    def get_cost_function(self):
        return theano.function(inputs=[self.question, self.question_mask, \
                                       self.answer, self.answer_mask], \
                               outputs=[self.cost], name='cost_function', \
                               on_unused_input='ignore')


    def get_deploy_function(self):
        maxout_input = tensor.concatenate([self.decoder_hidden_status[-1, :, :], \
                                           self.encoder_hidden_status[-1, :, :], \
                                           self.answer_embedding[-1, :, :]], \
                                          axis=1) 
        pred_word, pred_word_probability = \
                                    self.maxout_layer.get_output(base_data=self.tparams['Wemb_d'],
                                                                 refer_data=maxout_input)
        deploy_function = theano.function(inputs=[self.question, self.question_mask, \
                                                  self.answer, self.answer_mask], \
                                          outputs=[pred_word, pred_word_probability], \
                                          name='deploy_function')
        return deploy_function
    

    def get_observe_function(self):
        observe_function = theano.function(inputs=[self.question, self.question_mask], \
                                           outputs=[self.encoder_hidden_status[-1, :, :]], \
                                           name='observe_function')
        
        return observe_function
