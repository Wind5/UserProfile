import codecs
from test.test_audioop import datas



def import_dataset_list(path):
    datas_list = list()
    with codecs.open(path, 'r', 'gb18030') as fo:
        for line in fo.readlines():
            [user, age, gender, education], querys = \
                    line.strip().split('\t')[0:4], line.strip().split('\t')[4:]
            datas_list.append([user, age, gender, education, querys])
            
    return datas_list


def write_dataset_list(datas_list, col, path):
    with open(path, 'w') as fw:
        for idx, infos in enumerate(datas_list):
            print idx
            fw.writelines((infos[0] + '\t' + infos[col] + '\t' + '\t'.join(infos[4]) + '\n').encode('gb18030'))
    

datas_list = import_dataset_list('E://Github/UserProfile/data/origin_data/user_tag_query.2W.TRAIN.splitword')
write_dataset_list(datas_list, 1, 'E://Github/UserProfile/data/origin_data/user_tag_query.2W.AGE')
write_dataset_list(datas_list, 2, 'E://Github/UserProfile/data/origin_data/user_tag_query.2W.GENDER')
write_dataset_list(datas_list, 3, 'E://Github/UserProfile/data/origin_data/user_tag_query.2W.EDUCATION')