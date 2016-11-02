# -*- encoding=gb18030 -*-
import codecs
import numpy
import math



class Analysis:
    
    def __init__(self):
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
    
    
    def probability(self, age_list): 
        age_dict = dict()
        for age in age_list:
            if age not in age_dict:
                age_dict[age] = 0
            age_dict[age] += 1
        age_sum = sum(age_dict.values())
        for age in age_dict:
            age_dict[age] = 1.0 * age_dict[age] / age_sum 
        print age_dict
        
        return age_dict
    
    
    def analyze(self, train_info_path, test_info_path): 
        self.train_dataset_list = self.import_info_list(train_info_path, mode='train')
        self.probability([int(info[1]) for info in self.train_dataset_list])
        self.probability([int(info[2]) for info in self.train_dataset_list])
        self.probability([int(info[3]) for info in self.train_dataset_list])