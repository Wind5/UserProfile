from snownlp import SnowNLP
import codecs

def import_data(path):
	texts = list()
	with codecs.open(path, 'r', 'gb18030') as fo:
		for line in fo.readlines():
			[user, age, gender, education], querys = line.strip().split('\t')[0:4], line.strip().split('\t')[4:]
			texts.append([user, age, querys])

	return texts

def keyword(texts, path):
	user_list = list()
	word_dict = dict()
	idx = 0
	for user, age, querys in texts:
		idx += 1
		print idx
		s = SnowNLP('\t'.join(querys))
		keys = s.keywords(100)
		user_list.append(keys)
		for word in keys:
			if word not in word_dict:
				word_dict[word] = 0
			word_dict[word] += 1
	new_word_dict = dict()
	for idx, word in enumerate(sorted(word_dict.iteritems(), key=lambda x: x[1], reverse=True)[0:10000]):
		new_word_dict[word[0]] = idx
	print 'start write ...'
	with open(path, 'w') as fw:
		idx = 0
		for info, words in zip(texts, user_list):
			idx += 1
			print idx
			vector = [0]*len(new_word_dict)
			for word in words:
				if word in new_word_dict:
					vector[new_word_dict[word]] = 1
			fw.writelines((info[0] + '\t' + info[1] + '\t' + ' '.join([str(t) for t in vector]) + '\n').encode('gb18030'))


texts = import_data('E://Github/UserProfile/data/origin_data/user_tag_query.2W.TRAIN')
keyword(texts, 'E://Github/UserProfile/data/dataset_bow/train_dataset_list.AGE')