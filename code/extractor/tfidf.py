# -*- encoding=gb18030 -*-
import codecs
import numpy
import math



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
        with codecs.open(path, 'r', 'gb18030') as fo:
            for line in fo.readlines():
                [word, times, idx] = line.strip().split('\t')
                if word not in self.word2index:
                    self.word2index[word] = [times, idx]
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
        word_list = sorted(word_dict.iteritems(), key=lambda x: x[1], reverse=True)[0:5000]
        self.word2index = dict([(word[0], [word[1], idx]) for idx, word in enumerate(word_list)])
        self.index2word = dict([(idx, [word[0], word[1]]) for idx, word in enumerate(word_list)])
        
    
    def calculate_idf(self, info_list, mode='train'):
        for info in info_list:
            if mode == 'train':
                [user, age, gender, education, querys] = info
            else:
                [user, querys] = info
            idf_vector = [0]*len(self.word2index)
            word_list = list()
            for query in querys:
                for word, pos in query:
                    word_name = word + '<:>' + pos
                    if word_name in self.word2index:
                        word_list.append(word + '<:>' + pos)
            for word_name in set(word_list):
                idf_vector[self.word2index[word_name][1]] += 1
            for idx in range(len(idf_vector)):
                idf_vector[idx] = math.log(1.0 * len(info_list) / (idf_vector[idx]+1))
        
        return idf_vector
                    


    def construct_dataset(self, train_info_list, test_info_list):
        """
        Filter the word dict.
        Construct BOW dataset.
        """
        # construct train_dataset_list
        idf_vector = self.calculate_idf(train_info_list, mode='train')
        sparse_rate = 0.0
        vectors = list()
        for user, age, gender, education, querys in train_info_list:
            vector = [0] * len(self.word2index)
            for query in querys :
                for word, pos in query:
                    word_name = word + '<:>' + pos
                    if word_name in self.word2index:
                        vector[self.word2index[word_name][1]] += 1
            sparse_rate += 1.0 * sum([1 for t in vector if t != 0]) / len(self.word2index)
            vectors.append(vector)
        for idx, info in enumerate(train_info_list):
            new_vector = [0]*len(self.word2index)
            for word in range(len(vectors[idx])):
                word_sum = sum([vector[word] for vector in vectors]) + 1
                new_vector[word] = 1.0 * vectors[idx][word] * idf_vector[word] / word_sum 
            self.train_dataset_list.append((user, age, gender, education, new_vector))
        sparse_rate = 1.0 * sparse_rate / len(train_info_list)
        print 'training set sparse rate is', sparse_rate
        # construct test_dataset_list
        idf_vector = self.calculate_idf(test_info_list, mode='test')
        sparse_rate = 0.0
        vectors = list()
        for user, querys in test_info_list:
            vector = [0] * len(self.word2index)
            for query in querys :
                for word, pos in query:
                    word_name = word + '<:>' + pos
                    if word_name in self.word2index:
                        vector[self.word2index[word_name][1]] += 1
            sparse_rate += 1.0 * sum([1 for t in vector if t != 0]) / len(self.word2index)
            vectors.append(vector)
        for idx, info in enumerate(test_info_list):
            new_vector = [0]*len(self.word2index)
            for word in range(len(vectors[idx])):
                word_sum = sum([vector[word] for vector in vectors]) + 1
                new_vector[word] = 1.0 * vectors[idx][word] * idf_vector[word] / word_sum 
            self.test_dataset_list.append((user, new_vector))
        sparse_rate = 1.0 * sparse_rate / len(test_info_list)
        print 'training set sparse rate is', sparse_rate


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
            self.construct_word_dict(train_info_list, test_info_list)
            self.write_word_list(word_path)
        print len(self.word2index), len(self.index2word)
        self.construct_dataset(train_info_list, test_info_list)
        self.write_dataset_list(self.train_dataset_list, train_dataset_path, mode='train')
        self.write_dataset_list(self.test_dataset_list, test_dataset_path, mode='test')