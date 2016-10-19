import codecs
import numpy
from sklearn.cross_validation  import train_test_split
    


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
                    [user, age, gender, education], querys = line.strip().split('\t')[0:4], line.strip().split('\t')[4:]
                    if age != '0':
                        age_datas.append(querys)
                        age_labels.append(age)
                    if gender != '0':
                        gender_datas.append(querys)
                        gender_labels.append(gender)
                    if education != '0':
                        education_datas.append(querys)
                        education_labels.append(education)
                elif mode == 'test':
                    [user], querys = line.strip().split('\t')[0:1], line.strip().split('\t')[1:]
                    age_datas.append(querys)
                    gender_datas.append(querys)
                    education_datas.append(querys)
        age_datas = numpy.array(age_datas)
        gender_datas = numpy.array(gender_datas)
        education_datas = numpy.array(education_datas)
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
                
                
    def write_dataset_list(self, datas, labels, path):
        with open(path, 'w') as fw:
            for i in range(datas.shape[0]):
                fw.writelines((str(labels[i]) + '\t' + '\t'.join(datas[i]) + '\n').encode('gb18030'))
    


cls = Classifier()
cls.construct_dataset('train_dataset_list', 'test_dataset_list', mode='train')
cls.write_dataset_list(cls.train_age_datas, cls.train_age_labels, 'age.train')
cls.write_dataset_list(cls.test_age_datas, cls.test_age_labels, 'age.test')
cls.write_dataset_list(cls.train_gender_datas, cls.train_gender_labels, 'gender.train')
cls.write_dataset_list(cls.test_gender_datas, cls.test_gender_labels, 'gender.test')
cls.write_dataset_list(cls.train_education_datas, cls.train_education_labels, 'education.train')
cls.write_dataset_list(cls.test_education_datas, cls.test_education_labels, 'education.test')