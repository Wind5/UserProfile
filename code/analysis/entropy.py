# -*- encoding=gb18030 -*-
import codecs
import numpy
from sklearn.cross_validation  import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import entropy



class Analysis:
    
    def __init__(self):
        self.train_datas, self.test_datas = None, None
        self.train_age_labels, self.test_age_labels = None, None
        self.train_gender_labels, self.test_gender_labels = None, None
        self.train_education_labels, self.test_education_labels = None, None
    

    def import_dataset_list(self, path, mode='train'):
        """
        Import info_list and store as (user, age, gender, education, querys).
        """
        info_list, datas, age_labels, gender_labels, education_labels = list(), list(), list(), list(), list()
        with codecs.open(path, 'r', 'gb18030') as fo:
            for line in fo.readlines():
                if mode == 'train':
                    [user, age, gender, education, vector] = line.strip().split('\t')
                    if age == '0' or gender == '0' or education == '0' :
                        continue
                    age_labels.append(age)
                    gender_labels.append(gender)
                    education_labels.append(education)
                    vector = vector.split(' ')
                    datas.append(vector)
                elif mode == 'test':
                    [user, vector] = line.strip().split('\t')
                    vector = vector.split(' ')
                    datas.append(vector)
        datas = numpy.array(datas, dtype=int)
        print 'number of samples is', datas.shape[0]
        age_labels = numpy.array(age_labels, dtype=int)
        gender_labels = numpy.array(gender_labels, dtype=int)
        education_labels = numpy.array(education_labels, dtype=int)
        
        return datas, age_labels, gender_labels, education_labels
    
    
    def split_train_test(self, datas, age_labels, gender_labels, education_labels, \
                         test_size=0.2):
        """
        Split datas into (train, test) by test_size and train_size
        """
        train_size = 1 - test_size
        self.train_datas, self.test_datas, self.train_age_labels, self.test_age_labels = \
            train_test_split(datas, age_labels, test_size=test_size, train_size=train_size, \
                             random_state=False)
        _, _, self.train_gender_labels, self.test_gender_labels = \
            train_test_split(datas, gender_labels, test_size=test_size, train_size=train_size, \
                             random_state=False)
        _, _, self.train_education_labels, self.test_education_labels = \
            train_test_split(datas, education_labels, test_size=test_size, train_size=train_size, \
                             random_state=False)
    

    def construct_dataset(self, path, mode='train'):
        """
        Construct self.train_datas, self.test_datas and so on.
        """
        if mode == 'train':
            datas, age_labels, gender_labels, education_labels = \
                self.import_dataset_list(path, mode='train')
            self.split_train_test(datas, age_labels, gender_labels, education_labels, \
                                  test_size=0.2)
        elif mode == 'test':
            datas, age_labels, gender_labels, education_labels = \
                self.import_dataset_list(path, mode='train')
            self.train_datas, self.train_age_labels, self.train_gender_labels, self.train_education_labels = \
                datas, age_labels, gender_labels, education_labels
            datas, _, _, _ = self.import_dataset_list(path, mode='test')
            self.test_datas = datas
            
            
    def analyze1(self, path):
        self.construct_dataset(path, mode='train')
        print self.train_datas, self.train_gender_labels
        exp_entropy = entropy(self.train_gender_labels)
        ce_list = list()
        for word in range(self.train_datas.shape[1]):
            for label in set(self.train_gender_labels.tolist()):
                sub_array = numpy.array([self.train_datas[i,word] for i in range(self.train_datas.shape[0]) \
                                         if self.train_gender_labels[i] == label])
                min_entropy = min(min_entropy, 1.0 * sub_array.shape[0] * entropy(sub_array) / self.train_datas.shape[0])
            condition_entropy = min_entropy
            ce_list.append(condition_entropy)
        
        for idx, etp in sorted(enumerate(ce_list), key=lambda x: x[1], reverse=False)[0:10]:
            print idx, [self.train_datas[i,idx] for i in range(self.train_datas.shape[0]) \
                                         if self.train_gender_labels[i] == 1], \
                [self.train_datas[i,idx] for i in range(self.train_datas.shape[0]) \
                 if self.train_gender_labels[i] == 2]
                
                
    def analyze(self, path, index2word):
        self.construct_dataset(path, mode='train')
        pred_label = self.train_gender_labels
        print self.train_datas, pred_label
        exp_entropy = entropy(pred_label)
        n_young_sum = sum([len([1 for word in range(self.train_datas.shape[1]) if self.train_datas[i, word] != 0]) \
                           for i in range(self.train_datas.shape[0]) if pred_label[i] == 1])
        n_mid_sum = sum([len([1 for word in range(self.train_datas.shape[1]) if self.train_datas[i, word] != 0]) \
                           for i in range(self.train_datas.shape[0]) if pred_label[i] == 2])
        # n_old_sum = sum([len([1 for word in range(self.train_datas.shape[1]) if self.train_datas[i, word] != 0]) \
        #                    for i in range(self.train_datas.shape[0]) if pred_label[i] in [5,6]])
        print n_young_sum, n_mid_sum#, n_old_sum
        ce_list = list()
        for word in range(self.train_datas.shape[1]):
            min_entropy = 100
            n_young_search = 1.0 * len([1 for i in range(self.train_datas.shape[0]) \
                                        if (pred_label[i] == 1 and \
                                            self.train_datas[i, word] != 0)]) / n_young_sum
            n_mid_search = 1.0 * len([1 for i in range(self.train_datas.shape[0]) \
                                      if (pred_label[i] == 2 and \
                                          self.train_datas[i, word] != 0)]) / n_mid_sum
            # n_old_search = 1.0 * len([1 for i in range(self.train_datas.shape[0]) \
            #                           if (pred_label[i] in [5,6] and \
            #                               self.train_datas[i, word] != 0)]) / n_old_sum
            rate = entropy([n_young_search, n_mid_search])
            ce_list.append([word, rate, n_young_search, n_mid_search])
        
        for word, rate, n_young_search, n_mid_search in sorted(ce_list, key=lambda x: x[1], reverse=False):
            print index2word[word], rate, n_young_search, n_mid_search