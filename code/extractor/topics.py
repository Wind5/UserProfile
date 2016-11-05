# -*- encoding=gb18030 -*-
import codecs
import numpy
# import lda

from code.extractor.word_dict.construct_word_dict import WordDict



class Extractor:
    
    def __init__(self):
        self.word2index, self.index2word = dict(), dict()
        self.train_dataset_list, self.test_dataset_list = list(), list()
        self.train_dataset_array, self.test_dataset_array = list(), list()


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
                        vector[self.word2index[word_name][1]] += 1
            self.train_dataset_list.append((user, age, gender, education, vector))
            self.train_dataset_array.append(vector)
            sparse_rate += 1.0 * sum([1 for t in vector if t != 0]) / len(self.word2index)
        sparse_rate = 1.0 * sparse_rate / len(train_info_list)
        self.train_dataset_array = numpy.array(self.train_dataset_array)
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
            self.test_dataset_array.append(vector)
            sparse_rate += 1.0 * sum([1 for t in vector if t != 0]) / len(self.word2index)
        sparse_rate = 1.0 * sparse_rate / len(test_info_list)
        self.test_dataset_array = numpy.array(self.test_dataset_array)
        print 'testing set sparse rate is', sparse_rate
        
        
    def lda_topic_model(self, train_dataset_array, test_dataset_array):
        self.topic_words, self.train_doc_topics, self.test_doc_topics = list(), list(), list()
        dataset_array = numpy.concatenate([train_dataset_array, test_dataset_array], axis=0)
        model = lda.LDA(n_topics=100, n_iter=1000, random_state=1)
        model.fit(dataset_array)
        for i, word_dist in enumerate(model.topic_word_):
            sorted_list = sorted(enumerate(word_dist.tolist()), key=lambda x: x[1], reverse=True)
            self.topic_words.append([i, [self.index2word[word][0] for word, v in sorted_list]])
        for i, topic_dist in enumerate(model.doc_topic_):
            if i < train_dataset_array.shape[0]:
                info = self.train_dataset_list[i][0:4]
                self.train_doc_topics.append([info[0], info[1], info[2], info[3], topic_dist.tolist()])
            else:
                info = self.test_dataset_list[i-train_dataset_array.shape[0]][0:1]
                self.test_doc_topics.append([info[0], topic_dist.tolist()])


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


    def write_topic_words(self, topic_words, path):
        with open(path, 'w') as fw:
            for idx, words in topic_words:
                fw.writelines((str(idx) + '\t' + ' '.join(words) + '\n').encode('gb18030'))


    def write_word_list(self, path):
        with open(path, 'w') as fw:
            for idx in self.index2word:
                fw.writelines((self.index2word[idx][0] + '\t' + str(idx) + '\t' + \
                               str(self.index2word[idx][1]) + '\n').encode('gb18030'))
                    
    
    def extract(self, train_info_path, test_info_path, word_path, topic_word_path, \
                train_dataset_path, test_dataset_path, word_dict_existed=False):
        train_info_list = self.import_info_list(train_info_path, mode='train')
        test_info_list = self.import_info_list(test_info_path, mode='test')
        if word_dict_existed:
            self.import_word_dict(word_path)
        else:
            worddict = WordDict()
            self.word2index, self.index2word = \
                worddict.entropy_construct(train_info_list, test_info_list, size=5000)
            self.write_word_list(word_path)
        print len(self.word2index), len(self.index2word)
        self.construct_dataset(train_info_list, test_info_list)
        self.lda_topic_model(self.train_dataset_array, self.test_dataset_array)
        self.write_dataset_list(self.train_doc_topics, train_dataset_path, mode='train')
        self.write_dataset_list(self.test_doc_topics, test_dataset_path, mode='test')
        self.write_topic_words(self.topic_words, topic_word_path)