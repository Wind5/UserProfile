# -*- encoding=gb18030 -*-
import sys
import os
import codecs
import numpy

from code.analysis.label import Analysis
from code.extractor.single.bow import Extractor
from code.classification.single.decision_tree import Classifier



if __name__ == '__main__':
    base_path = os.path.join(os.getcwd(), 'data')
    dataset_folder = os.path.join(base_path, sys.argv[1])
    # dataset_folder2 = os.path.join(base_path, sys.argv[2])
    label = sys.argv[2]
    '''
    dataset_folder2 = os.path.join(base_path, sys.argv[2])
    trainset_file = os.path.join(base_path, sys.argv[2] + '.train')
    validset_file = os.path.join(base_path, sys.argv[2] + '.valid')
    testset_file = os.path.join(base_path, sys.argv[2] + '.test')
    prset_file = os.path.join(base_path, sys.argv[2] + '.pr')
    dict_file = os.path.join(base_path, sys.argv[3])
    algo_name = sys.argv[4]
    '''
    mode = sys.argv[3]
    print ('Now dataset is %s, label is %s, mode is %s.' % (dataset_folder, label, mode))
    
    # model selection
    '''
    if algo_name == 'HierarchicalEncoder' :
        from deep.manage.model.hierarchical_encoder import HierarchicalEncoderModel
        manager = HierarchicalEncoderModel(dataset_folder, trainset_file, testset_file, \
                                           dict_file, algo_name, mode)
    '''
        
    # mode selection
    if mode == 'extract':
        extractor = Extractor()
        extractor.extract(os.path.join(dataset_folder, 'user_tag_query.2W.'+label) , \
                          os.path.join(dataset_folder, 'user_tag_query.2W.TEST.splitword'), \
                          os.path.join(dataset_folder, 'word_dict.'+label+'.txt'), \
                          # os.path.join(dataset_folder, 'topic_word.txt'), \
                          os.path.join(dataset_folder, 'train_dataset_list.'+label), \
                          os.path.join(dataset_folder, 'test_dataset_list.'+label))
    elif mode == 'classify':
        clf = Classifier()
        clf.classify(os.path.join(dataset_folder, 'train_dataset_list.'+label), \
                     os.path.join(dataset_folder, 'test_dataset_list.'+label), \
                     # os.path.join(dataset_folder2, 'train_dataset_list'), \
                     # os.path.join(dataset_folder2, 'test_dataset_list'), \
                     os.path.join(dataset_folder, 'pred_labels.'+label), mode='train')
    elif mode == 'analyze':
        analysis = Analysis()
        analysis.analyze(os.path.join(dataset_folder1, 'user_tag_query.2W.TRAIN.splitword') , \
                         os.path.join(dataset_folder1, 'user_tag_query.2W.TEST.splitword'))
    elif mode == 'deeptrain':
        manager.train()