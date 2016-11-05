# -*- encoding=gb18030 -*-
import codecs
import numpy

from code.extractor.word_dict.construct_word_dict import WordDict



class Extractor:
    
    def __init__(self):
        self.word2index, self.index2word = dict(), dict()
        self.train_dataset_list, self.test_dataset_list = list(), list()


    def import_info_list(self, path, mode='train'):
        """
        Import info_list and store as (user, label, querys).
        """
        info_list = list()
        with codecs.open(path, 'r', 'gb18030') as fo:
            for line in fo.readlines():
                new_querys = list()
                infos = line.strip().split('\t')
                if mode == 'train' :
                    [user, label], querys = infos[0:2], infos[2:]
                    for query in querys:
                        new_query = list()
                        for word_name in query.split(' '):
                            word, pos = word_name.split('<:>')
                            new_query.append((word, pos))
                        new_querys.append(new_query)
                    info_list.append((user, label, new_querys))
                elif mode=='test' :
                    [user], querys = infos[0:1], infos[1:]
                    for query in querys:
                        new_query = list()
                        for word_name in query.split(' '):
                            word, pos = word_name.split('<:>')
                            new_query.append((word, pos))
                        new_querys.append(new_query)
                    info_list.append((user, new_querys))
            
        return info_list
    
    
    def import_word_dict(self, path):
        with codecs.open(path, 'r', 'gb18030') as fo:
            for line in fo.readlines():
                [word, times, idx] = line.strip().split('\t')
                if word not in self.word2index:
                    self.word2index[word] = [times, idx]
                    self.index2word[idx] = word
                    idx += 1


    def construct_dataset(self, train_info_list, test_info_list):
        """
        Filter the word dict.
        Construct BOW dataset.
        """
        sparse_rate = 0.0
        for user, label, querys in train_info_list:
            vector = [0] * len(self.word2index)
            for query in querys :
                for word, pos in query:
                    word_name = word + '<:>' + pos
                    if word_name in self.word2index:
                        vector[self.word2index[word_name][1]] += 1
            self.train_dataset_list.append((user, label, vector))
            sparse_rate += 1.0 * sum([1 for t in vector if t != 0]) / len(self.word2index)
        sparse_rate = 1.0 * sparse_rate / len(train_info_list)
        print 'training set sparse rate is', sparse_rate
        # construct test_dataset_list
        sparse_rate = 0.0
        for user, querys in test_info_list:
            vector = [0] * len(self.word2index)
            for query in querys :
                for word, pos in query:
                    word_name = word + '<:>' + pos
                    if word_name in self.word2index:
                        vector[self.word2index[word_name][1]] += 1
            self.test_dataset_list.append((user, vector))
            sparse_rate += 1.0 * sum([1 for t in vector if t != 0]) / len(self.word2index)
        sparse_rate = 1.0 * sparse_rate / len(test_info_list)
        print 'testing set sparse rate is', sparse_rate


    def write_dataset_list(self, dataset_list, path, mode='train'):
        with open(path, 'w') as fw:
            for info in dataset_list:
                if mode == 'train':
                    user, label, vector = info
                    fw.writelines((user + '\t' + label + '\t' + \
                                   ' '.join([str(t) for t in vector]) + '\n').encode('gb18030'))
                elif mode == 'test':
                    user, vector = info
                    fw.writelines((user + '\t' + ' '.join([str(t) for t in vector]) + '\n').encode('gb18030'))


    def write_word_list(self, path):
        with open(path, 'w') as fw:
            for idx in self.index2word:
                fw.writelines((self.index2word[idx][0] + '\t' + str(idx) + '\t' + \
                               str(self.index2word[idx][1]) + '\n').encode('gb18030'))
                    
    
    def extract(self, train_info_path, test_info_path, word_path, \
                train_dataset_path, test_dataset_path, word_dict_existed=False):
        train_info_list = self.import_info_list(train_info_path, mode='train')
        test_info_list = self.import_info_list(test_info_path, mode='test')
        if word_dict_existed:
            self.import_word_dict(word_path)
        else:
            worddict = WordDict()
            self.word2index, self.index2word = \
                worddict.entropy_single_construct(train_info_list, test_info_list, size=10000)
            self.write_word_list(word_path)
        print len(self.word2index), len(self.index2word)
        self.construct_dataset(train_info_list, test_info_list)
        self.write_dataset_list(self.train_dataset_list, train_dataset_path, mode='train')
        self.write_dataset_list(self.test_dataset_list, test_dataset_path, mode='test')