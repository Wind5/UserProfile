# -*- encoding=gb18030 -*-
import sys
import os
import codecs
import numpy
from pyltp import Segmentor, Postagger
from sklearn.cross_validation  import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from code.analysis.entropy import Analysis
from code.extractor.tfidf import Extractor
from code.classification.decision_tree import Classifier



if __name__ == '__main__':
    base_path = os.path.join(os.getcwd(), 'data')
    dataset_folder = os.path.join(base_path, sys.argv[1]) 
    trainset_file = os.path.join(base_path, sys.argv[2] + '.train')
    validset_file = os.path.join(base_path, sys.argv[2] + '.valid')
    testset_file = os.path.join(base_path, sys.argv[2] + '.test')
    prset_file = os.path.join(base_path, sys.argv[2] + '.pr')
    dict_file = os.path.join(base_path, sys.argv[3])
    algo_name = sys.argv[4]
    mode = sys.argv[5]
    print ('Now use %s model to %s the dataset %s.' % (algo_name, mode, sys.argv[1]))
    
    # model selection
    if algo_name == 'HierarchicalEncoder' :
        from deep.manage.model.hierarchical_encoder import HierarchicalEncoderModel
        manager = HierarchicalEncoderModel(dataset_folder, trainset_file, testset_file, \
                                           dict_file, algo_name, mode)
        
    # mode selection
    if mode == 'extract':
        extractor = Extractor()
        extractor.extract(os.path.join(dataset_folder, 'user_tag_query.2W.TRAIN.splitword') , \
                          os.path.join(dataset_folder, 'user_tag_query.2W.TEST.splitword'), \
                          os.path.join(dataset_folder, 'word_dict.txt'), \
                          os.path.join(dataset_folder, 'train_dataset_list'), \
                          os.path.join(dataset_folder, 'test_dataset_list'))
    elif mode == 'analyze':
        analysis = Analysis()
        word2index, index2word = import_word_dict('data/word_dict')
        analysis.analyze('data/train_dataset_list', index2word)
    elif mode == 'classify':
        clf = Classifier()
        clf.classify(os.path.join(dataset_folder, 'train_dataset_list'), \
                     os.path.join(dataset_folder, 'test_dataset_list'), \
                     os.path.join(dataset_folder, 'pred_labels'), mode='train')    
    elif mode == 'deeptrain':
        manager.train()