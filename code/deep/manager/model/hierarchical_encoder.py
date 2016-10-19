# -*- encoding = gb18030 -*-
import os
import re
import codecs
import math
import numpy
import heapq
from collections import OrderedDict

from deep.manage.manager import ModelManager
from deep.util.parameter_operation import *
from deep.algorithms.networks.hierarchical_encoder import HierarchicalEncoderNetwork
from deep.dataloader.corpus_reader_querys_label import CorpusReaderQuerysLabel



class HierarchicalEncoderModel(ModelManager) :
    
    def __init__(self, dataset_folder, trainset_file, testset_file, dict_file, algo_name, mode) :
        """
        Need to set these attributes.
            1. conf_dict: configuration of the model.
            2. cr: CorpursReader for operate data.
            3. model: the network model.
        """
        self.conf_dict = OrderedDict({'algo_name': algo_name, 'save_freq': 5})
        hyper_params = OrderedDict({'batch_size': 5, 'word_embedding_dim': 2, 'hidden_dim': 3})
        self.conf_dict.update(hyper_params)
        if mode == 'train' :
            self.param_path = os.path.join(dataset_folder, 'model', 'dialog', \
                                           get_params_file_name(self.conf_dict) + '.model')
        else :
            self.param_path = os.path.join(dataset_folder, 'model', 'dialog', \
                                           get_params_file_name(self.conf_dict) + '.model.final')
        self.param_dict = load_params_val(self.param_path)
        self.conf_path = os.path.join(dataset_folder, 'model', 'dialog', \
                                      get_params_file_name(self.conf_dict) + '.conf')
        save_confs_val(self.conf_dict, self.conf_path)
        # set corpus reader
        if mode == 'train' :
            self.cr = CorpusReaderQuerysLabel(trainset_file=trainset_file, \
                                              testset_file=testset_file, \
                                              dict_file=dict_file, \
                                              is_BEG_available=False, \
                                              is_END_available=True)
        elif mode == 'generate' :
            self.cr = CorpusReaderQuerysLabel(trainset_file=None, \
                                              testset_file=None, \
                                              dict_file=dict_file, \
                                              is_BEG_available=False, \
                                              is_END_available=True)
        # set model
        self.model = HierarchicalEncoderNetwork(n_words=len(self.cr.get_word_dictionary()), \
                                                hidden_status_dim=self.conf_dict['hidden_dim'], \
                                                word_embedding_dim=self.conf_dict['word_embedding_dim'], \
                                                input_params=self.param_dict)