# -*- encoding=gb18030 -*-
import codecs
import numpy
from sklearn.cross_validation  import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC



class Classifier:
    
    def __init__(self):
        self.train_age_datas, self.test_age_datas = None, None
        self.train_age_labels, self.test_age_labels = None, None
        self.train_gender_datas, self.test_gender_datas = None, None
        self.train_gender_labels, self.test_gender_labels = None, None
        self.train_education_datas, self.test_education_datas = None, None
        self.train_education_labels, self.test_education_labels = None, None
    

    def import_dataset_list(self, path, mode='train'):
        """
        Import info_list and store as (user, age, gender, education, querys).
        """
        info_list = list()
        age_datas, age_labels = list(), list()
        gender_datas, gender_labels = list(), list()
        education_datas, education_labels = list(), list()
        with codecs.open(path, 'r', 'gb18030') as fo:
            for line in fo.readlines():
                if mode == 'train':
                    [user, age, gender, education, vector] = line.strip().split('\t')
                    vector = vector.split(' ')
                    if age != '0':
                        age_datas.append(vector)
                        age_labels.append(age)
                    if gender != '0':
                        gender_datas.append(vector)
                        gender_labels.append(gender)
                    if education != '0':
                        education_datas.append(vector)
                        education_labels.append(education)
                elif mode == 'test':
                    [user, vector] = line.strip().split('\t')
                    vector = vector.split(' ')
                    age_datas.append(vector)
                    gender_datas.append(vector)
                    education_datas.append(vector)
        age_datas = numpy.array(age_datas, dtype=int)
        gender_datas = numpy.array(gender_datas, dtype=int)
        education_datas = numpy.array(education_datas, dtype=int)
        print 'number of age_datas samples is', age_datas.shape[0]
        print 'number of gender_datas samples is', gender_datas.shape[0]
        print 'number of education_datas samples is', education_datas.shape[0]
        age_labels = numpy.array(age_labels, dtype=int)
        gender_labels = numpy.array(gender_labels, dtype=int)
        education_labels = numpy.array(education_labels, dtype=int)
        
        return age_datas, age_labels, gender_datas, gender_labels, education_datas, education_labels
    
    
    def split_train_test(self, age_datas, age_labels, gender_datas, gender_labels, \
                         education_datas, education_labels, test_size=0.2):
        """
        Split datas into (train, test) by test_size and train_size
        """
        train_size = 1 - test_size
        self.train_age_datas, self.test_age_datas, self.train_age_labels, self.test_age_labels = \
            train_test_split(age_datas, age_labels, test_size=test_size, train_size=train_size, \
                             random_state=False)
        self.train_gender_datas, self.test_gender_datas, self.train_gender_labels, self.test_gender_labels = \
            train_test_split(gender_datas, gender_labels, test_size=test_size, train_size=train_size, \
                             random_state=False)
        self.train_education_datas, self.test_education_datas, self.train_education_labels, self.test_education_labels = \
            train_test_split(education_datas, education_labels, test_size=test_size, train_size=train_size, \
                             random_state=False)
    

    def construct_dataset(self, train_path, test_path, mode='train'):
        """
        Construct self.train_datas, self.test_datas and so on.
        """
        if mode == 'train':
            age_datas, age_labels, gender_datas, gender_labels, education_datas, education_labels = \
                self.import_dataset_list(train_path, mode='train')
            self.split_train_test(age_datas, age_labels, gender_datas, gender_labels, \
                                  education_datas, education_labels, test_size=0.2)
        elif mode == 'test':
            age_datas, age_labels, gender_datas, gender_labels, education_datas, education_labels = \
                self.import_dataset_list(train_path, mode='train')
            self.train_age_datas, self.train_age_labels, self.train_gender_datas, self.train_gender_labels, \
            self.train_education_datas, self.train_education_labels = \
                age_datas, age_labels, gender_datas, gender_labels, education_datas, education_labels
            self.test_age_datas, _, self.test_gender_datas, _, self.test_education_datas, _ = \
                self.import_dataset_list(test_path, mode='test')
        
        
    def classification(self, datas_train, datas_test, age_labels_train, age_labels_test):
        clf = self.train_model(datas_train, age_labels_train)
        age_labels_pred = self.predict_model(clf, datas_test)
        acc = 1.0 * sum([1 for idx in range(age_labels_test.shape[0]) \
                         if age_labels_pred[idx] == age_labels_test[idx]]) / age_labels_test.shape[0]
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
            _, age_acc = self.classification(self.train_age_datas, self.test_age_datas, \
                                             self.train_age_labels, self.test_age_labels)
            _, gender_acc = self.classification(self.train_gender_datas, self.test_gender_datas, \
                                                self.train_gender_labels, self.test_gender_labels)
            _, education_acc = self.classification(self.train_education_datas, self.test_education_datas, \
                                                   self.train_education_labels, self.test_education_labels)
            print (age_acc + gender_acc + education_acc) / 3
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