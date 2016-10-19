# -*- encoding = gb18030 -*-
import os 
import codecs
import numpy
from multiprocessing.dummy import Pool as ThreadPool

from deep.dataloader.util import get_mask_data
from deep.dataloader.corpus_reader import CorpusReader 
import deep.util.config as config 

    
    
class CorpusReaderQuerysLabel(CorpusReader):
    
    def __init__(self, trainset_file, testset_file, dict_file, \
                 is_BEG_available=True, is_END_available=True):
        
        CorpusReader.__init__(self, trainset_file=trainset_file, testset_file=testset_file, \
                              dict_file=dict_file, \
                              is_BEG_available=is_BEG_available, is_END_available=is_END_available)
    

    def load_data(self, dataset_file, charset='g18030'):
        dialogs = list()
        with codecs.open(dataset_file, 'r', charset) as fo:
            for line in fo.readlines() :
                    [post, response] = line.strip().split('\t')
                    post = self.load_sentence(post.split(' '), \
                                              self.word2index, special_flag=self.special_flag)
                    response = self.load_sentence(response.split(' '), \
                                                  self.word2index, special_flag=self.special_flag)
                    dialogs.append([post, response])
            dialogs = [[post[:-1], [post[-1]] + response] for post, response in dialogs]
                
        print ('Number of dataset: %i' % (len(dialogs)))
        return dialogs
    
    
    def load_sentence(self, sentence, word2index, special_flag) :
        index_sentence = list()
        for word in sentence :
            if word in word2index :
                index_sentence.append(word2index[word])
            else :
                index_sentence.append(word2index['<UNK>'])
        if '<BEG>' in special_flag :
            index_sentence = [word2index['<BEG>']] + index_sentence
        if '<END>' in special_flag :
            index_sentence = index_sentence + [word2index['<END>']]
            
        return index_sentence
    
    
    def get_model_input(self, scope, dialogs):
        batch_question = [dialog[0] for dialog in dialogs[scope[0]:scope[1]]]
        batch_answer = [dialog[1] for dialog in dialogs[scope[0]:scope[1]]]
        if batch_question != [] :
            question, question_mask, answer, answer_mask = \
                self.get_mask(batch_question, batch_answer)
        else :
            question, question_mask, answer, answer_mask = None, None, None, None
        return question, question_mask, numpy.transpose(answer), numpy.transpose(answer_mask)
 
 
    def get_mask(self, question, answer) :
        q, a = question, answer
        question_dims, answer_dims = [0,0], [0,0]
        question_dims[0] = len(question)
        answer_dims[0] = len(answer)
        question_dims[1] = max_length = 30
        answer_dims[1] = max([len(s) for s in answer])
        question = numpy.zeros((question_dims[0], question_dims[1]), dtype='int64')
        question_mask = numpy.zeros((question_dims[0], question_dims[1]), \
                                    dtype=config.globalFloatType())
        answer = numpy.zeros((answer_dims[0], answer_dims[1]), dtype='int64')
        answer_mask = numpy.zeros((answer_dims[0], answer_dims[1]), \
                                  dtype=config.globalFloatType())
        for x, stc in enumerate(q) :
            for y, s in enumerate(stc[0:max_length]) :
                question[x,y] = s
                question_mask[x,y] = 1.0
        for x, stc in enumerate(a) :
            for y, s in enumerate(stc) :
                answer[x,y] = s
                answer_mask[x,y] = 1.0
        return question, question_mask, answer, answer_mask
    
  
    def transform_input_data(self, sentence):
        sentence = self.load_sentence(sentence, self.word2index, special_flag=self.special_flag)
        print sentence
        question, question_mask, _, _ = self.get_mask([sentence], [list()])
        return question, question_mask
