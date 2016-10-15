# -*- encoding=gb18030 -*-
import codecs
import numpy



class Extractor:
    
    def __init__(self):
        self.word2index, self.index2word = dict(), dict()
        self.train_dataset_list, self.test_dataset_list = list(), list()


    def import_info_list(self, path, mode='train'):
        """
        Import info_list and store as (user, age, gender, education, querys).
        """
        info_list = list()
        with codecs.open(path, 'r', 'gb18030') as fo:
            for line in fo.readlines():
                new_querys = list()
                infos = line.strip().split('\t')
                if mode == 'train' :
                    [user, age, gender, education], querys = infos[0:4], infos[4:]
                    for query in querys:
                        new_query = list()
                        for word_name in query.split(' '):
                            word, pos = word_name.split('<:>')
                            new_query.append((word, pos))
                        new_querys.append(new_query)
                    info_list.append((user, age, gender, education, new_querys))
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
        idx = 0
        with codecs.open(path, 'r', 'gb18030') as fo:
            for line in fo.readlines():
                [word, _, _] = line.strip().split('\t')
                if word not in self.word2index:
                    self.word2index[word] = idx
                    self.index2word[idx] = word
                    idx += 1
    
    
    def construct_word_dict(self, train_info_list, test_info_list):
        """
        Construct word_dict with train and test.
        """
        word_dict, train_dataset_list, test_dataset_list, sparse_rate = dict(), list(), list(), 0.0
        for user, age, gender, education, querys in train_info_list:
            for query in querys:
                for word, pos in query:
                    word_name = word + '<:>' + pos
                    if pos in ['n', 'nh', 'ni', 'nl', 'ns', 'nz']:
                        if word_name not in word_dict:
                            word_dict[word_name] = 0
                        word_dict[word_name] += 1
        for user, querys in test_info_list:
            for query in querys:
                for word, pos in query:
                    word_name = word + '<:>' + pos
                    if pos in ['n', 'nh', 'ni', 'nl', 'ns', 'nz']:
                        if word_name not in word_dict:
                            word_dict[word_name] = 0
                        word_dict[word_name] += 1
        word_list = sorted(word_dict.iteritems(), key=lambda x: x[1], reverse=True)[0:2000]
        word2index = dict([(word, idx) for idx, word in enumerate([t[0] for t in word_list])])


    def construct_dataset(self, train_info_list, test_info_list):
        """
        Filter the word dict.
        Construct BOW dataset.
        """
        sparse_rate = 0.0
        for user, age, gender, education, querys in train_info_list:
            vector = [0] * len(self.word2index)
            for query in querys :
                for word, pos in query:
                    word_name = word + '<:>' + pos
                    if word_name in self.word2index:
                        vector[self.word2index[word_name]] += 1
            self.train_dataset_list.append((user, age, gender, education, vector))
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
                        vector[self.word2index[word_name]] += 1
            self.test_dataset_list.append((user, vector))
            sparse_rate += 1.0 * sum([1 for t in vector if t != 0]) / len(self.word2index)
        sparse_rate = 1.0 * sparse_rate / len(test_info_list)
        print 'testing set sparse rate is', sparse_rate


    def write_dataset_list(self, dataset_list, path, mode='train'):
        with open(path, 'w') as fw:
            for info in dataset_list:
                if mode == 'train':
                    user, age, gender, education, vector = info
                    fw.writelines((user + '\t' + age + '\t' + gender + '\t' + education +\
                                   '\t' + ' '.join([str(t) for t in vector]) + '\n').encode('gb18030'))
                elif mode == 'test':
                    user, vector = info
                    fw.writelines((user + '\t' + ' '.join([str(t) for t in vector]) + '\n').encode('gb18030'))
                    
    
    def extract(self, train_info_path, test_info_path, word_path, \
                train_dataset_path, test_dataset_path):
        self.import_word_dict(word_path)
        print len(self.word2index), len(self.index2word)
        train_info_list = self.import_info_list(train_info_path, mode='train')
        test_info_list = self.import_info_list(test_info_path, mode='test')
        self.construct_dataset(train_info_list, test_info_list)
        self.write_dataset_list(self.train_dataset_list, train_dataset_path, mode='train')
        self.write_dataset_list(self.test_dataset_list, test_dataset_path, mode='test')