# -*- encoding=gb18030 -*-
import codecs
import numpy
from sklearn.cross_validation  import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC



class Classifier:
    
    def __init__(self, param_dict):
        self.train_datas, self.test_datas = None, None
        self.train_labels, self.test_labels = None, None
        self.param_dict = param_dict
    

    def import_dataset_list(self, path, mode='train'):
        """
        Import info_list and store as (user, label, querys).
        """
        info_list = list()
        datas_list, labels_list = list(), list()
        with codecs.open(path, 'r', 'gb18030') as fo:
            for line in fo.readlines():
                if mode == 'train':
                    [user, label, vector] = line.strip().split('\t')
                    vector = vector.split(' ')
                    if label != '0':
                        datas_list.append(vector)
                        labels_list.append(label)
                elif mode == 'test':
                    [user, vector] = line.strip().split('\t')
                    vector = vector.split(' ')
                    datas_list.append(vector)
        datas_list = numpy.array(datas_list, dtype=float)
        print 'number of datas_list samples is', datas_list.shape[0]
        labels_list = numpy.array(labels_list, dtype=int)
        
        return datas_list, labels_list
    
    
    def split_train_test(self, datas_list, labels_list, test_size=0.2):
        """
        Split datas into (train, test) by test_size and train_size
        """
        train_size = 1 - test_size
        self.train_datas, self.test_datas, self.train_labels, self.test_labels = \
            train_test_split(datas_list, labels_list, test_size=test_size, train_size=train_size, \
                             random_state=False)
    

    def construct_dataset(self, train_path, test_path, mode='train'):
        """
        Construct self.train_datas, self.test_datas and so on.
        """
        if mode == 'train':
            datas_list, labels_list = self.import_dataset_list(train_path, mode='train')
            self.split_train_test(datas_list, labels_list, test_size=0.2)
        elif mode == 'test':
            self.train_datas, self.train_labels = self.import_dataset_list(train_path, mode='train')
        
        
    def classification(self, datas_train, datas_test, age_labels_train, age_labels_test):
        clf = self.train_model(datas_train, age_labels_train)
        age_labels_pred = self.predict_model(clf, datas_test)
        acc = 0.0
        for idx in range(age_labels_test.shape[0]):
            if age_labels_pred[idx] == age_labels_test[idx]:
                acc += 1.0
        acc /= age_labels_test.shape[0]
        print 'accurancy is ', acc
        return age_labels_pred, acc


    def train_model(self, datas_train, age_labels_train):
        print 'number of estimators is ', self.param_dict['n_estimators']
        clf = RandomForestClassifier(n_estimators=self.param_dict['n_estimators'])
        clf.fit(datas_train, age_labels_train)
        
        return clf


    def predict_model(self, clf, datas_test):
        labels_pred = clf.predict(datas_test)
        
        return labels_pred
                
                
    def write_labels_pred(self, labels_pred, path):
        with open(path, 'w') as fw:
            for idx in range(labels_pred.shape[0]):
                fw.writelines((str(labels_pred[idx]) + '\n').encode('gb18030'))
                
                
    def classify(self, train_dataset_path, test_dataset_path, pred_label_path, mode='train'):
        if mode == 'train':
            self.construct_dataset(train_dataset_path, test_dataset_path, mode=mode)
            _, acc = self.classification(self.train_datas, self.test_datas, \
                                         self.train_labels, self.test_labels)
        if mode == 'test':
            self.construct_dataset(train_dataset_path, test_dataset_path, mode=mode)
            model = self.train_model(self.train_datas, self.train_labels)
            print 'finish train age_model ...'
            labels_pred = self.predict_model(model, test_datas)
            self.write_labels_pred(labels_pred, pred_label_path)