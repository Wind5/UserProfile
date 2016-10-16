# -*- encoding=gb18030 -*-
import sys
import codecs
import numpy
from pyltp import Segmentor, Postagger
from sklearn.cross_validation  import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from code.analysis.entropy import Analysis
from code.extractor.bow import Extractor
from code.classification.decision_tree import Classifier



if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'extract':
        extractor = Extractor()
        extractor.extract('data/user_tag_query.top.TRAIN.splitword', \
                          'data/user_tag_query.top.TEST.splitword', 'data/word_dict', \
                          'data/train_dataset_list', 'data/test_dataset_list', \
                          word_dict_existed=False)
    elif mode == 'analyze':
        analysis = Analysis()
        word2index, index2word = import_word_dict('data/word_dict')
        analysis.analyze('data/train_dataset_list', index2word)
    elif mode == 'classify':
        clf = Classifier()
        clf.classify('data/train_dataset_list', 'data/test_dataset_list', \
                     'data/labels_pred', mode='test')