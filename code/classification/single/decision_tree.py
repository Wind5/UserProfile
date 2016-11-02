# -*- encoding=gb18030 -*-
import codecs
import numpy
from sklearn.cross_validation  import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC



class Classifier:
    
    def __init__(self):
        self.train_datas, self.test_datas = None, None
        self.train_labels, self.test_labels = None, None
    

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
            datas_list, labels_list = self.import_dataset_list(train_path, mode='train')
            self.train_age_datas, self.train_age_labels, self.train_gender_datas, self.train_gender_labels, \
            self.train_education_datas, self.train_education_labels = \
                age_datas, age_labels, gender_datas, gender_labels, education_datas, education_labels
            self.test_age_datas, _, self.test_gender_datas, _, self.test_education_datas, _ = \
                self.import_dataset_list(test_path, mode='test')
        
        
    def classification(self, datas_train, datas_test, age_labels_train, age_labels_test):
        clf = self.train_model(datas_train, age_labels_train)
        age_labels_pred = self.predict_model(clf, datas_test)
        acc = 0.0
        for idx in range(age_labels_test.shape[0]):
            if age_labels_pred[idx] == age_labels_test[idx]:
                acc += 1.0
        acc /= age_labels_test.shape[0]
        print acc
        return age_labels_pred, acc


    def train_model(self, datas_train, age_labels_train):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(datas_train, age_labels_train)
        
        return clf


    def predict_model(self, clf, datas_test):
        age_labels_pred = clf.predict(datas_test)
        
        return age_labels_pred
                
                
    def write_labels_pred(self, age_labels_pred, gender_labels_pred, education_labels_pred, path):
        with open(path, 'w') as fw:
            for idx in range(age_labels_pred.shape[0]):
                fw.writelines((str(age_labels_pred[idx]) + ' ' + \
                               str(gender_labels_pred[idx]) + ' ' + \
                               str(education_labels_pred[idx]) + '\n').encode('gb18030'))
                
                
    def classify(self, train_dataset_path, test_dataset_path, pred_label_path, mode='train'):
        if mode == 'train':
            self.construct_dataset(train_dataset_path, test_dataset_path, mode=mode)
            _, acc = self.classification(self.train_datas, self.test_datas, \
                                         self.train_labels, self.test_labels)
        if mode == 'test':
            self.construct_dataset(train_dataset_path, test_dataset_path, mode=mode)
            age_model = self.train_model(self.train_age_datas, self.train_age_labels)
            print 'finish train age_model ...'
            age_labels_pred = self.predict_model(age_model, self.test_age_datas)
            gender_model = self.train_model(self.train_gender_datas, self.train_gender_labels)
            print 'finish train gender_model ...'
            gender_labels_pred = self.predict_model(gender_model, self.test_gender_datas)
            education_model = self.train_model(self.train_education_datas, self.train_education_labels)
            print 'finish train education_model ...'
            education_labels_pred = self.predict_model(education_model, self.test_education_datas)
            self.write_labels_pred(age_labels_pred, gender_labels_pred, education_labels_pred, \
                                   pred_label_path)