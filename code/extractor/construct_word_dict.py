import numpy
    

def frequency_construct(train_info_list, test_info_list):
    """
    Construct word_dict with train and test.
    """
    word_dict  = dict()
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
    word_list = sorted(word_dict.iteritems(), key=lambda x: x[1], reverse=True)[0:5000]
    word2index = dict([(word[0], [word[1], idx]) for idx, word in enumerate(word_list)])
    index2word = dict([(idx, [word[0], word[1]]) for idx, word in enumerate(word_list)])
    
    return word2index, index2word