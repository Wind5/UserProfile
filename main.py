# -*- encoding=gb18030 -*-
import sys
import os
import codecs
import numpy

from code.analysis.label import Analysis
from code.extractor.single.bow import Extractor
from code.classification.single.decision_tree import Classifier



def process(extractor_params, classifier_params, mode):
    base_path = os.path.join(os.getcwd(), 'data')
    dataset_folder = os.path.join(base_path, sys.argv[1])
    label = sys.argv[2]
    mode = sys.argv[3]
    # mode selection
    if mode == 'extract':
        extractor = Extractor(extractor_params)
        extractor.extract(os.path.join(dataset_folder, 'user_tag_query.2W.'+label) , \
                          os.path.join(dataset_folder, 'user_tag_query.2W.TEST.splitword'), \
                          os.path.join(dataset_folder, 'word_dict.'+label+'.txt'), \
                          os.path.join(dataset_folder, 'train_dataset_list.'+label), \
                          os.path.join(dataset_folder, 'test_dataset_list.'+label))
    elif mode == 'classify':
        clf = Classifier(classifier_params)
        clf.classify(os.path.join(dataset_folder, 'train_dataset_list.'+label), \
                     os.path.join(dataset_folder, 'test_dataset_list.'+label), \
                     os.path.join(dataset_folder, 'pred_labels.'+label), mode='test')
    elif mode == 'analyze':
        analysis = Analysis()
        analysis.analyze(os.path.join(dataset_folder1, 'user_tag_query.2W.TRAIN.splitword') , \
                         os.path.join(dataset_folder1, 'user_tag_query.2W.TEST.splitword'))
        
        
def main():
    for n_words in [3000, 4000]:#, 5000, 6000, 8000, 10000, 12000, 15000]:
        for n_estimators in [100, 200]:#, 300, 500]:
            extractor_params = {'n_words': n_words}
            classifier_params = {'n_estimators': n_estimators}
            process(extractor_params, classifier_params, mode='extract')
            process(extractor_params, classifier_params, mode='classify')
            print 'finish process ' + '#'*30 
            
            
if __name__ == '__main__':
    main()