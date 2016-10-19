# coding:gb18030
import codecs
import math
import heapq
import string
import numpy as np
from abc import ABCMeta, abstractmethod

from deep.util.parameter_operation import save_params_val, load_params_val
import deep.util.config as config 
from deep.manage.algorithm import beam_search, greed



class ModelManager :
    __metaclass__ = ABCMeta
    
    def __init__(self):
        """
        Different init in sub class. 
        """
        
        
    def print_model_info(self):
        print ('model configuration is shown below ...')
        for key in self.conf_dict:
            print ('%s is %s' % (key, self.conf_dict[key]))
        print ('Saving parameters to %s.' % (self.param_path))
        
    
    def train(self):
        """
        Train a model.
        """
        n_trainset, n_validset, n_testset = self.cr.get_size()
        n_batches = max(1, (n_trainset - 1) / self.conf_dict['batch_size'] + 1)
        # print model information
        print '#' * 100
        self.print_model_info()
        print ('Compiling model...')
        train_model = \
            self.model.get_training_function(self.cr, batch_size=self.conf_dict['batch_size'])
        valid_model = self.model.get_validing_function(self.cr, batch_size=self.conf_dict['batch_size'])
        test_model = self.model.get_testing_function(self.cr, batch_size=self.conf_dict['batch_size'])
        # start train
        print '#' * 100
        print ('Start to train.') 
        test_error = test_model()[0]
        print ('Now testing model. Cost Error: %.10f' % (test_error))
        epoch, it, n_epochs = 0, 0, 1000000
        while (epoch < n_epochs):
            epoch += 1
            for i in xrange(n_batches):
                # train model
                train_error = train_model(i)[0]
                # print 'Step error: %f\r' % train_error,
                if math.isnan(train_error):
                    print ('Train error is NaN in iteration %d, batch %d' % (epoch, i))
                    error_model_path = self.param_path + str(epoch) + '.error' 
                    save_params_val(error_model_path, self.model.get_parameters())
                    print ('model saved in %s , reload and skip the %d batch.' % \
                        (error_model_path, self.conf_dict['save_freq']))
                    param_dict = load_params_val(self.param_path)
                    self.model.set_parameters(param_dict)
                    exit()
                if it % self.conf_dict['save_freq'] == 0:
                    # valid model
                    # valid_error = valid_model()[0]
                    print ('@iter: %d\tTraining Error: %.10f' % (it, train_error))
                    print self.param_path
                    save_params_val(self.param_path, self.model.get_parameters())
                it = it + 1
            # test model
            # print ('Finished a epoch.')
            test_error = test_model()[0]
            print ('Now testing model. Testing Error: %.10f' % (test_error))
            save_params_val(self.param_path + str(epoch), self.model.get_parameters())
            
            
    def generate(self, input_file, output_file):
        """
        Generate a model.
        """
        deploy_model = self.model.get_deploy_function()
        with open(output_file, 'w') as fw:
            with codecs.open(input_file, 'r', config.globalCharSet()) as fo:
                for line in fo.readlines() :
                    # line_word, line_zi = SegProcess(line.strip())
                    # line = line_zi.decode("gb18030")
                    line = line.strip()
                    print (line.encode(config.globalCharSet()))
                    fw.writelines('%s\n' % line.encode(config.globalCharSet()))
                    
                    res, score = self.generate_one_question(line, deploy_model,
                                                            output_size=5,
                                                            beam_size=100,
                                                            search_scope=100)
                    for r, s in zip(res, score) :
                        print ('result: %s, score: %f' % (r.encode(config.globalCharSet()), s))
                        fw.writelines('result: %s, score: %f' % (r.encode(config.globalCharSet()), s))
                    fw.writelines('\n')