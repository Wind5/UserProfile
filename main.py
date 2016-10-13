# -*- encoding=gb18030 -*-
import codecs
import numpy
from pyltp import Segmentor, Postagger
from sklearn.cross_validation  import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



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


def import_info_list(path, mode='train'):
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
                user, querys = infos[0:1], infos[1:]
                for query in querys:
                    new_query = list()
                    for word_name in query.split(' '):
                        word, pos = word_name.split('<:>')
                        new_query.append((word, pos))
                    new_querys.append(new_query)
                info_list.append((user, new_querys))
        
    return info_list


def import_dataset_list(path):
    """
    Import info_list and store as (user, age, gender, education, querys).
    """
    info_list, datas, age_labels, gender_labels, education_labels = list(), list(), list(), list(), list()
    with codecs.open(path, 'r', 'gb18030') as fo:
        for line in fo.readlines():
            [user, age, gender, education, vector] = line.strip().split('\t')
            age_labels.append(age)
            gender_labels.append(gender)
            education_labels.append(education)
            vector = vector.split(' ')
            datas.append(vector)
    datas = numpy.array(datas, dtype=int)
    age_labels = numpy.array(age_labels, dtype=int)
    gender_labels = numpy.array(gender_labels, dtype=int)
    education_labels = numpy.array(education_labels, dtype=int)
    datas_train, datas_test, age_labels_train, age_labels_test = \
        train_test_split(datas, age_labels, test_size=0.2, train_size=0.8, random_state=False)
    datas_train, datas_test, gender_labels_train, gender_labels_test = \
        train_test_split(datas, gender_labels, test_size=0.2, train_size=0.8, random_state=False)
    datas_train, datas_test, education_labels_train, education_labels_test = \
        train_test_split(datas, education_labels, test_size=0.2, train_size=0.8, random_state=False)
    return (datas_train, datas_test), (age_labels_train, age_labels_test), \
        (gender_labels_train, gender_labels_test), (education_labels_train, education_labels_test)


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


def construct_dataset(train_info_list, test_info_list, mode='train'):
    """
    Filter the word dict.
    Construct BOW dataset.
    """
    word_dict, train_dataset_list, test_dataset_list, sparse_rate = dict(), list(), list(), 0.0
    for user, age, gender, education, querys in train_info_list:
        for query in querys:
            for word, pos in query:
                word_name = word + '<:>' + pos
                if pos in ['n', 'nh', 'ni', 'nl', 'ns', 'nz']:
                    if word_name not in word_dict:
                        word_dict[word_name] = 0
                    word_dict[word_name] += 1
    for user, querys in test_info_list:
        for query in querys:
            for word, pos in query:
                word_name = word + '<:>' + pos
                if pos in ['n', 'nh', 'ni', 'nl', 'ns', 'nz']:
                    if word_name not in word_dict:
                        word_dict[word_name] = 0
                    word_dict[word_name] += 1
    word_list = sorted(word_dict.iteritems(), key=lambda x: x[1], reverse=True)[0:2000]
    word2index = dict([(word, idx) for idx, word in enumerate([t[0] for t in word_list])])
    # construct train_dataset_list
    for user, age, gender, education, querys in train_info_list:
        vector = [0] * len(word2index)
        for query in querys :
            for word, pos in query:
                word_name = word + '<:>' + pos
                if word_name in word2index:
                    vector[word2index[word_name]] += 1
            train_dataset_list.append((user, age, gender, education, vector))
        sparse_rate += 1.0 * sum([1 for t in vector if t != 0]) / len(word2index)
    sparse_rate = 1.0 * sparse_rate / len(info_list)
    print 'training set sparse rate is', sparse_rate
    # construct test_dataset_list
    sparse_rate = 0.0
    for user, querys in test_info_list:
        vector = [0] * len(word2index)
        for query in querys :
            for word, pos in query:
                word_name = word + '<:>' + pos
                if word_name in word2index:
                    vector[word2index[word_name]] += 1
            test_dataset_list.append((user, vector))
        sparse_rate += 1.0 * sum([1 for t in vector if t != 0]) / len(word2index)
    sparse_rate = 1.0 * sparse_rate / len(info_list)
    print 'testing set sparse rate is', sparse_rate
    
    return word_list, train_dataset_list, test_dataset_list


def write_word_list(word_list, path):
    with open(path, 'w') as fw:
        for idx, word in enumerate(word_list):
            fw.writelines((word[0] + '\t' + str(word[1]) + '\t' + str(idx) + '\n').encode('gb18030'))


def write_dataset_list(dataset_list, path, mode='train'):
    with open(path, 'w') as fw:
        for info in dataset_list:
            if mode == 'train':
                user, age, gender, education, vector = info
                fw.writelines((user + '\t' + age + '\t' + gender + '\t' + education +\
                               '\t' + ' '.join([str(t) for t in vector]) + '\n').encode('gb18030'))
            elif mode == 'test':
                user, vector = info
                fw.writelines((user + '\t' + ' '.join([str(t) for t in vector]) + '\n').encode('gb18030'))
        
        
def classification(datas_train, datas_test, age_labels_train, age_labels_test):
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(datas_train, age_labels_train)
    age_labels_pred = clf.predict(datas_test)
    acc = 1.0 * sum([1 for idx in range(age_labels_test.shape[0]) \
                     if age_labels_pred[idx] == age_labels_test[idx]]) / age_labels_test.shape[0]
    print acc
    return acc
                
                
                
# info_list = import_data('E://Github/CCF_Competition/data/user_tag_query.2W.TEST', mode='test')
# ltp_labeling(info_list, 'E://Github/CCF_Competition/data/user_tag_query.2W.TEST.splitword', mode='test')
train_info_list = import_info_list('data/user_tag_query.2W.TRAIN.splitword', mode='train')
test_info_list = import_info_list('data/user_tag_query.2W.TEST.splitword', mode='test')
print 'start construct_dataset ...'
word_list, train_dataset_list, test_dataset_list = construct_dataset(train_info_list, test_info_list)
print 'finish construct_dataset ...'
write_dataset_list(train_dataset_list, 'data/train_dataset_list', mode='train')
write_dataset_list(test_dataset_list, 'data/test_dataset_list', mode='test')
write_word_list(word_list, 'data/word_dict')
'''
(datas_train, datas_test), (age_labels_train, age_labels_test), \
    (gender_labels_train, gender_labels_test), (education_labels_train, education_labels_test) = \
    import_dataset_list('data/dataset_list')
age_acc = classification(datas_train, datas_test, age_labels_train, age_labels_test)
gender_acc = classification(datas_train, datas_test, gender_labels_train, gender_labels_test)
education_acc = classification(datas_train, datas_test, education_labels_train, education_labels_test)
print (age_acc + gender_acc + education_acc) / 3
'''