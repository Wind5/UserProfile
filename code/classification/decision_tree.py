# -*- encoding=gb18030 -*-
import codecs
import numpy
from sklearn.cross_validation  import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC



class Classifier:
    
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
    

    def construct_dataset(self, train_path, test_path, mode='train'):
        """
        Construct self.train_datas, self.test_datas and so on.
        """
        if mode == 'train':
            datas, age_labels, gender_labels, education_labels = \
                self.import_dataset_list(train_path, mode='train')
            self.split_train_test(datas, age_labels, gender_labels, education_labels, \
                                  test_size=0.2)
        elif mode == 'test':
            datas, age_labels, gender_labels, education_labels = \
                self.import_dataset_list(train_path, mode='train')
            self.train_datas, self.train_age_labels, self.train_gender_labels, self.train_education_labels = \
                datas, age_labels, gender_labels, education_labels
            datas, _, _, _ = self.import_dataset_list(test_path, mode='test')
            self.test_datas = datas
        
        
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
            _, age_acc = self.classification(self.train_datas, self.test_datas, \
                                             self.train_age_labels, self.test_age_labels)
            _, gender_acc = self.classification(self.train_datas, self.test_datas, \
                                                self.train_gender_labels, self.test_gender_labels)
            _, education_acc = self.classification(self.train_datas, self.test_datas, \
                                                   self.train_education_labels, self.test_education_labels)
            print (age_acc + gender_acc + education_acc) / 3
        if mode == 'test':
            self.construct_dataset(train_dataset_path, test_dataset_path, mode=mode)
            age_model = self.train_model(self.train_datas, self.train_age_labels)
            print 'finish train age_model ...'
            gender_model = self.train_model(self.train_datas, self.train_gender_labels)
            print 'finish train gender_model ...'
            education_model = self.train_model(self.train_datas, self.train_education_labels)
            print 'finish train education_model ...'
            age_labels_pred = self.predict_model(age_model, self.test_datas)
            gender_labels_pred = self.predict_model(gender_model, self.test_datas)
            education_labels_pred = self.predict_model(education_model, self.test_datas)
            self.write_labels_pred(age_labels_pred, gender_labels_pred, education_labels_pred, \
                                   pred_label_path)