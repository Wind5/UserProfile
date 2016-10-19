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
                        for zi in query.split(' '):
                            new_query.append(zi)
                        new_querys.append(new_query)
                    info_list.append((user, age, gender, education, new_querys))
                elif mode=='test' :
                    [user], querys = infos[0:1], infos[1:]
                    for query in querys:
                        new_query = list()
                        for zi in query.split(' '):
                            new_query.append(zi)
                        new_querys.append(new_query)
                    info_list.append((user, new_querys))
            
        return info_list
    
    
    def import_word_dict(self, path):
        with codecs.open(path, 'r', 'gb18030') as fo:
            for line in fo.readlines():
                [word, times, idx] = line.strip('\n').split('\t')
                if word not in self.word2index:
                    self.word2index[word] = [times, idx]
                    self.index2word[idx] = word


    def construct_dataset(self, train_info_list, test_info_list):
        """
        Filter the word dict.
        Construct BOW dataset.
        """
        # construct train_dataset_list
        for user, age, gender, education, querys in train_info_list:
            new_querys = list()
            for query in querys :
                new_query = list()
                for zi in query:
                    if zi in self.word2index:
                        new_query.append(self.word2index[zi][1])
                new_querys.append(new_query)
            self.train_dataset_list.append((user, age, gender, education, new_querys))
        # construct test_dataset_list
        for user, querys in test_info_list:
            new_querys = list()
            for query in querys :
                new_query = list()
                for zi in query:
                    if zi in self.word2index:
                        new_query.append(self.word2index[zi][1])
                new_querys.append(new_query)
            self.test_dataset_list.append((user, new_querys))


    def write_dataset_list(self, dataset_list, path, mode='train'):
        with open(path, 'w') as fw:
            for info in dataset_list:
                if mode == 'train':
                    user, age, gender, education, querys = info
                    querys_str = '\t'.join([' '.join(query) for query in querys])
                    fw.writelines((user + '\t' + age + '\t' + gender + '\t' + education +\
                                   '\t' + querys_str + '\n').encode('gb18030'))
                elif mode == 'test':
                    user, querys = info
                    querys_str = '\t'.join([' '.join(query) for query in querys])
                    fw.writelines((user + '\t' + querys_str + '\n').encode('gb18030'))
                    
    
    def extract(self, train_info_path, test_info_path, word_path, \
                train_dataset_path, test_dataset_path):
        train_info_list = self.import_info_list(train_info_path, mode='train')
        test_info_list = self.import_info_list(test_info_path, mode='test')
        self.import_word_dict(word_path)
        print len(self.word2index), len(self.index2word)
        self.construct_dataset(train_info_list, test_info_list)
        self.write_dataset_list(self.train_dataset_list, train_dataset_path, mode='train')
        self.write_dataset_list(self.test_dataset_list, test_dataset_path, mode='test')