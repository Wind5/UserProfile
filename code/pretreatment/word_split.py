# -*- encoding=gb18030 -*-
import codecs
import numpy
from pyltp import Segmentor, Postagger



def import_data(path, mode='train'):
    """
    Import dataset and store as (user, age, gender, education, querys).
    """
    info_list = list()
    with codecs.open(path, 'r', 'gb18030') as fo:
        for line in fo.readlines():
            infos = line.strip().split('\t')
            if mode == 'train' :
                [user, age, gender, education], querys = infos[0:4], infos[4:]
                info_list.append((user, age, gender, education, querys))
            elif mode == 'test' :
                [user], querys = infos[0:1], infos[1:]
                info_list.append((user, querys))
    
    return info_list


def ltp_labeling(info_list, path, mode='train'):
    """
    Split word in querys and write in the disk.
    """
    new_info_list, word_dict, dataset_list, sparse_rate = list(), dict(), list(), 0.0
    segmentor = Segmentor()
    segmentor.load('E://Github/CCF_Competition/data/ltp_data/cws.model')
    postagger = Postagger()
    postagger.load('E://Github/CCF_Competition/data/ltp_data/pos.model')
    idx = 0
    for info in info_list:
        if idx % 100 == 0 :
            print 1.0 * idx / len(info_list)
        idx += 1
        if mode == 'train' :
            user, age, gender, education, querys = info
        elif mode == 'test' :
            user, querys = info
        new_querys = list()
        for query in querys:
            new_query = list()
            words = segmentor.segment(query.encode('utf8'))
            postags = postagger.postag(words)
            for word, pos in zip(words, postags):
                word, pos = word.decode('utf8'), pos.decode('utf8')
                new_query.append((word, pos))
            new_querys.append(new_query)
        if mode == 'train' :
            new_info_list.append((user, age, gender, education, new_querys))
        elif mode == 'test' :
            new_info_list.append((user, new_querys))
    # write in the disk
    with open(path, 'w') as fw:
        for info in new_info_list:
            if mode == 'train' :
                user, age, gender, education, querys = info
                query_str = '\t'.join([' '.join([word+'<:>'+pos for word, pos in query]) for query in querys])
                fw.writelines((user + '\t' + age + '\t' + gender + '\t' + education +\
                               '\t' + query_str + '\n').encode('gb18030'))
            elif mode == 'test' :
                user, querys = info
                query_str = '\t'.join([' '.join([word+'<:>'+pos for word, pos in query]) for query in querys])
                print user, query_str
                fw.writelines((user + '\t' + query_str + '\n').encode('gb18030'))   
                
                
def main():
    train_info_list = import_data('E://Github/CCF_Competition/origin_data/user_tag_query.2W.TRAIN', mode='train')
    print 'finish importing train_info_list ...'
    ltp_labeling(train_info_list, 'E://Github/CCF_Competition/data/user_tag_query.2W.TRAIN.splitword', mode='train')
    print 'finish spliting words in train_info_list ...'
    test_info_list = import_data('E://Github/CCF_Competition/origin_data/user_tag_query.2W.TEST', mode='test')
    print 'finish importing test_info_list ...'
    ltp_labeling(test_info_list, 'E://Github/CCF_Competition/data/user_tag_query.2W.TEST.splitword', mode='test')
    print 'finish spliting words in test_info_list ...'
    

    
####################################################################################################
main()