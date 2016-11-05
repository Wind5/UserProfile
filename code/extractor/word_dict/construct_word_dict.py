import numpy
from scipy.stats import entropy
    


class WordDict:
    
    def frequency_construct(self, train_info_list, test_info_list, size=5000):
        """
        Construct word_dict with train and test.
        """
        word_dict  = dict()
        for user, age, gender, education, querys in train_info_list:
            for query in querys:
                for word, pos in query:
                    word_name = word + '<:>' + pos
                    if pos in ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'ws', 'v', 'j']:
                        if word_name not in word_dict:
                            word_dict[word_name] = 0
                        word_dict[word_name] += 1
        for user, querys in test_info_list:
            for query in querys:
                for word, pos in query:
                    word_name = word + '<:>' + pos
                    if pos in ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'ws', 'v', 'j']:
                        if word_name not in word_dict:
                            word_dict[word_name] = 0
                        word_dict[word_name] += 1
        word_list = sorted(word_dict.iteritems(), key=lambda x: x[1], reverse=True)[0:size]
        word2index = dict([(word[0], [word[1], idx]) for idx, word in enumerate(word_list)])
        index2word = dict([(idx, [word[0], word[1]]) for idx, word in enumerate(word_list)])
        
        return word2index, index2word
    
    
    def entropy_construct(self, train_info_list, test_info_list, size=5000):
        """
        Construct word_dict with train and test.
        sort word by (alpha * term_frequency + entropy) 
        """
        entropy_dict, freqency_dict, word_dict = dict(), dict(), dict()
        for user, age, gender, education, querys in train_info_list:
            word_list = list()
            for query in querys:
                for word, pos in query:
                    if pos in ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'ws', 'v', 'j']:
                        word_name = word + '<:>' + pos
                        word_list.append(word_name)
            for word_name in set(word_list):
                if word_name not in entropy_dict:
                    entropy_dict[word_name] = [[0]*6, [0]*2, [0]*6]
                entropy_dict[word_name][0][int(age)-1] += 1
                entropy_dict[word_name][1][int(gender)-1] += 1
                entropy_dict[word_name][2][int(education)-1] += 1
                if word_name not in freqency_dict:
                    freqency_dict[word_name] = 0
                freqency_dict[word_name] += 1
        print 'finish statistic ...'
        for word_name in entropy_dict:
            etp1 = entropy([1.0/6]*6) - entropy(entropy_dict[word_name][0])
            etp2 = entropy([1.0/2]*2) - entropy(entropy_dict[word_name][1])
            etp3 = entropy([1.0/6]*6) - entropy(entropy_dict[word_name][2])
            etp = etp1 + etp2 + etp3 
            freq = 1.0 * freqency_dict[word_name] / len(freqency_dict)
            word_dict[word_name] = [etp, freq, etp*freq, entropy_dict[word_name][0]]
        word_list = sorted(word_dict.iteritems(), key=lambda x: x[1][2], reverse=True)[0:size]
        word2index = dict([(word[0], [word[1][2], idx]) for idx, word in enumerate(word_list)])
        index2word = dict([(idx, [word[0], word[1][2]]) for idx, word in enumerate(word_list)])
        
        return word2index, index2word
    
    
    def entropy_single_construct(self, train_info_list, test_info_list, size=5000):
        """
        Construct word_dict with train and test.
        sort word by (alpha * term_frequency + entropy) 
        """
        entropy_dict, freqency_dict, word_dict = dict(), dict(), dict()
        n_labels = len(set([label for user, label, querys in train_info_list]))
        for user, label, querys in train_info_list:
            word_list = list()
            for query in querys:
                for word, pos in query:
                    if pos in ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'v', 'ws', 'j']:
                        word_name = word + '<:>' + pos
                        word_list.append(word_name)
            for word_name in set(word_list):
                if word_name not in entropy_dict:
                    entropy_dict[word_name] = [0]*n_labels
                entropy_dict[word_name][int(label)-1] += 1
                if word_name not in freqency_dict:
                    freqency_dict[word_name] = 0
                freqency_dict[word_name] += 1
        print 'finish statistic ...'
        for word_name in entropy_dict:
            etp = entropy([1.0/n_labels]*n_labels) - entropy(entropy_dict[word_name])
            freq = 1.0 * freqency_dict[word_name] / len(freqency_dict)
            word_dict[word_name] = [etp, freq, etp*freq, entropy_dict[word_name]]
        word_list = sorted(word_dict.iteritems(), key=lambda x: x[1][2], reverse=True)[0:size]
        word2index = dict([(word[0], [word[1][2], idx]) for idx, word in enumerate(word_list)])
        index2word = dict([(idx, [word[0], word[1][2]]) for idx, word in enumerate(word_list)])
        
        return word2index, index2word