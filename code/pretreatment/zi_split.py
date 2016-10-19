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
        for line in fo.readlines()[0:10]:
            infos = line.strip().split('\t')
            if mode == 'train' :
                [user, age, gender, education], querys = infos[0:4], infos[4:]
                info_list.append((user, age, gender, education, querys))
            elif mode == 'test' :
                [user], querys = infos[0:1], infos[1:]
                info_list.append((user, querys))
    
    return info_list


def zi_spliting(info_list, path, mode='train'):
    """
    Split zi in querys and write in the disk.
    """
    new_info_list = list()
    for infos in info_list:
        if mode == 'train':
            [user, age, gender, education, querys] = infos
        elif mode == 'test':
            [user, querys] = infos
        new_querys = list()
        for query in querys:
            new_query = list()
            for zi in query:
                new_query.append(zi)
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
                query_str = '\t'.join([' '.join(query) for query in querys])
                fw.writelines((user + '\t' + age + '\t' + gender + '\t' + education +\
                               '\t' + query_str + '\n').encode('gb18030'))
            elif mode == 'test' :
                user, querys = info
                query_str = '\t'.join([' '.join(query) for query in querys])
                fw.writelines((user + '\t' + query_str + '\n').encode('gb18030')) 
    
    
def construct_zi_dict(train_info_list, test_info_list):
    """
    Construct zi_dict with train and test.
    """
    zi_dict, train_dataset_list, test_dataset_list = dict(), list(), list()
    for user, age, gender, education, querys in train_info_list:
        for query in querys:
            for zi in query:
                if zi not in zi_dict:
                    zi_dict[zi] = 0
                zi_dict[zi] += 1
    for user, querys in test_info_list:
        for query in querys:
            for zi in query:
                if zi not in zi_dict:
                    zi_dict[zi] = 0
                zi_dict[zi] += 1
    zi_list = sorted(zi_dict.iteritems(), key=lambda x: x[1], reverse=True)
    print zi_list
    zi2index = dict([(zi[0], [zi[1], idx]) for idx, zi in enumerate([t[0] for t in zi_list])])
    
    return zi2index


def write_word_list(zi2index, path):
    with open(path, 'w') as fw:
        for zi in zi2index:
            fw.writelines((zi + '\t' + str(zi2index[zi][0]) + '\t' \
                           + str(zi2index[zi][0]) + '\n').encode('gb18030'))
                
                
def main():
    train_info_list = import_data('E://Github/CCF_Competition/data/origin_data/user_tag_query.2W.TRAIN', mode='train')
    print 'finish importing train_info_list ...'
    zi_spliting(train_info_list, 'E://Github/CCF_Competition/data/user_tag_query.2W.TRAIN.splitzi', mode='train')
    print 'finish spliting zis in train_info_list ...'
    test_info_list = import_data('E://Github/CCF_Competition/data/origin_data/user_tag_query.2W.TEST', mode='test')
    print 'finish importing test_info_list ...'
    zi_spliting(test_info_list, 'E://Github/CCF_Competition/data/user_tag_query.2W.TEST.splitzi', mode='test')
    print 'finish spliting zis in test_info_list ...'
    zi2index = construct_zi_dict(train_info_list, test_info_list)
    print 'finish construct zi_dict in test_info_list ...'
    write_word_list(zi2index, 'E://Github/CCF_Competition/data/word_dict')
    

    
####################################################################################################
main()