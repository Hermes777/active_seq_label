
import nltk
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from nltk.tag.util import untag

 
tagged_sentences = nltk.corpus.treebank.tagged_sents()

train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))



def features(sentence, index):
	""" sentence: [w1, w2, ...], index: the index of the word """
	return {
	'word': sentence[index],
	'is_first': index == 0,
	'is_last': index == len(sentence) - 1,
	'is_capitalized': sentence[index][0].upper() == sentence[index][0],
	'is_all_caps': sentence[index].upper() == sentence[index],
	'is_all_lower': sentence[index].lower() == sentence[index],
	'prefix-1': sentence[index][0],
	'prefix-2': sentence[index][:2],
	'prefix-3': sentence[index][:3],
	'suffix-1': sentence[index][-1],
	'suffix-2': sentence[index][-2:],
	'suffix-3': sentence[index][-3:],
	'prev_word': '' if index == 0 else sentence[index - 1],
	'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
	'has_hyphen': '-' in sentence[index],
	'is_numeric': sentence[index].isdigit(),
	'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
	}
def word2features(sent, i):
	word = sent[i][0]
	postag = sent[i][1]

	features = {
		'bias': 1.0,
		'word.lower()': word.lower(),
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'prefix-1': word[0],
		'prefix-2': word[:2],
		'prefix-3': word[:3],
		'suffix-1': word[-1],
		'suffix-2': word[-2:],
		'suffix-3': word[-3:],
		'has_hyphen': '-' in word,
		'is_numeric': word.isdigit(),
		'capitals_inside': word[1:].lower() != word[1:]
	}
	if i > 0:
		word1 = sent[i-1][0]
		postag1 = sent[i-1][1]
		features.update({
			'-1:word.lower()': word1.lower(),
			'-1:word.istitle()': word1.istitle(),
			'-1:word.isupper()': word1.isupper(),
			'-1:prefix-1': word1[0],
			'-1:prefix-2': word1[:2],
			'-1:prefix-3': word1[:3],
			'-1:suffix-1': word1[-1],
			'-1:suffix-2': word1[-2:],
			'-1:suffix-3': word1[-3:],
			'-1:has_hyphen': '-' in word1,
			'-1:is_numeric': word1.isdigit(),
			'-1:capitals_inside': word1[1:].lower() != word1[1:]
		})
	else:
		features['BOS'] = True

	if i < len(sent)-1:
		word1 = sent[i+1][0]
		postag1 = sent[i+1][1]
		features.update({
			'+1:word.lower()': word1.lower(),
			'+1:word.istitle()': word1.istitle(),
			'+1:word.isupper()': word1.isupper(),
			'+1:prefix-1': word1[0],
			'+1:prefix-2': word1[:2],
			'+1:prefix-3': word1[:3],
			'+1:suffix-1': word1[-1],
			'+1:suffix-2': word1[-2:],
			'+1:suffix-3': word1[-3:],
			'+1:has_hyphen': '-' in word1,
			'+1:is_numeric': word1.isdigit(),
			'+1:capitals_inside': word1[1:].lower() != word1[1:]
		})
	else:
		features['EOS'] = True

	return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]
crf = CRF(
	algorithm='lbfgs',
	c1=0.1,
	c2=0.1,
	max_iterations=100,
	all_possible_transitions=True
)
crf.fit(X_train, y_train)
labels = list(crf.classes_)
labels.remove('O')
X_test=[[{'is_first': True, 'word': u'-', 'is_all_lower': True, 'prefix-3': u'-', 'is_capitalized': True, 'prefix-1': u'-', 'suffix-3': u'-', 'has_hyphen': True, 'is_numeric': False, 'next_word': u'Barrington', 'is_last': False, 'suffix-1': u'-', 'suffix-2': u'-', 'is_all_caps': True, 'capitals_inside': False, 'prefix-2': u'-', 'prev_word': ''}, {'is_first': False, 'word': u'Barrington', 'is_all_lower': False, 'prefix-3': u'Bar', 'is_capitalized': True, 'prefix-1': u'B', 'suffix-3': u'ton', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'Research', 'is_last': False, 'suffix-1': u'n', 'suffix-2': u'on', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'Ba', 'prev_word': u'-'}, {'is_first': False, 'word': u'Research', 'is_all_lower': False, 'prefix-3': u'Res', 'is_capitalized': True, 'prefix-1': u'R', 'suffix-3': u'rch', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'Associates', 'is_last': False, 'suffix-1': u'h', 'suffix-2': u'ch', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'Re', 'prev_word': u'Barrington'}, {'is_first': False, 'word': u'Associates', 'is_all_lower': False, 'prefix-3': u'Ass', 'is_capitalized': True, 'prefix-1': u'A', 'suffix-3': u'tes', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'Inc', 'is_last': False, 'suffix-1': u's', 'suffix-2': u'es', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'As', 'prev_word': u'Research'}, {'is_first': False, 'word': u'Inc', 'is_all_lower': False, 'prefix-3': u'Inc', 'is_capitalized': True, 'prefix-1': u'I', 'suffix-3': u'Inc', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'said', 'is_last': False, 'suffix-1': u'c', 'suffix-2': u'nc', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'In', 'prev_word': u'Associates'}, {'is_first': False, 'word': u'said', 'is_all_lower': True, 'prefix-3': u'sai', 'is_capitalized': False, 'prefix-1': u's', 'suffix-3': u'aid', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'Friday', 'is_last': False, 'suffix-1': u'd', 'suffix-2': u'id', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'sa', 'prev_word': u'Inc'}, {'is_first': False, 'word': u'Friday', 'is_all_lower': False, 'prefix-3': u'Fri', 'is_capitalized': True, 'prefix-1': u'F', 'suffix-3': u'day', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'it', 'is_last': False, 'suffix-1': u'y', 'suffix-2': u'ay', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'Fr', 'prev_word': u'said'}, {'is_first': False, 'word': u'it', 'is_all_lower': True, 'prefix-3': u'it', 'is_capitalized': False, 'prefix-1': u'i', 'suffix-3': u'it', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'upgraded', 'is_last': False, 'suffix-1': u't', 'suffix-2': u'it', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'it', 'prev_word': u'Friday'}, {'is_first': False, 'word': u'upgraded', 'is_all_lower': True, 'prefix-3': u'upg', 'is_capitalized': False, 'prefix-1': u'u', 'suffix-3': u'ded', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'Unitog', 'is_last': False, 'suffix-1': u'd', 'suffix-2': u'ed', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'up', 'prev_word': u'it'}, {'is_first': False, 'word': u'Unitog', 'is_all_lower': False, 'prefix-3': u'Uni', 'is_capitalized': True, 'prefix-1': u'U', 'suffix-3': u'tog', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'Co', 'is_last': False, 'suffix-1': u'g', 'suffix-2': u'og', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'Un', 'prev_word': u'upgraded'}, {'is_first': False, 'word': u'Co', 'is_all_lower': False, 'prefix-3': u'Co', 'is_capitalized': True, 'prefix-1': u'C', 'suffix-3': u'Co', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'to', 'is_last': False, 'suffix-1': u'o', 'suffix-2': u'Co', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'Co', 'prev_word': u'Unitog'}, {'is_first': False, 'word': u'to', 'is_all_lower': True, 'prefix-3': u'to', 'is_capitalized': False, 'prefix-1': u't', 'suffix-3': u'to', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'a', 'is_last': False, 'suffix-1': u'o', 'suffix-2': u'to', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'to', 'prev_word': u'Co'}, {'is_first': False, 'word': u'a', 'is_all_lower': True, 'prefix-3': u'a', 'is_capitalized': False, 'prefix-1': u'a', 'suffix-3': u'a', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'near-term', 'is_last': False, 'suffix-1': u'a', 'suffix-2': u'a', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'a', 'prev_word': u'to'}, {'is_first': False, 'word': u'near-term', 'is_all_lower': True, 'prefix-3': u'nea', 'is_capitalized': False, 'prefix-1': u'n', 'suffix-3': u'erm', 'has_hyphen': True, 'is_numeric': False, 'next_word': u'outperform', 'is_last': False, 'suffix-1': u'm', 'suffix-2': u'rm', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'ne', 'prev_word': u'a'}, {'is_first': False, 'word': u'outperform', 'is_all_lower': True, 'prefix-3': u'out', 'is_capitalized': False, 'prefix-1': u'o', 'suffix-3': u'orm', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'from', 'is_last': False, 'suffix-1': u'm', 'suffix-2': u'rm', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'ou', 'prev_word': u'near-term'}, {'is_first': False, 'word': u'from', 'is_all_lower': True, 'prefix-3': u'fro', 'is_capitalized': False, 'prefix-1': u'f', 'suffix-3': u'rom', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'a', 'is_last': False, 'suffix-1': u'm', 'suffix-2': u'om', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'fr', 'prev_word': u'outperform'}, {'is_first': False, 'word': u'a', 'is_all_lower': True, 'prefix-3': u'a', 'is_capitalized': False, 'prefix-1': u'a', 'suffix-3': u'a', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'long-term', 'is_last': False, 'suffix-1': u'a', 'suffix-2': u'a', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'a', 'prev_word': u'from'}, {'is_first': False, 'word': u'long-term', 'is_all_lower': True, 'prefix-3': u'lon', 'is_capitalized': False, 'prefix-1': u'l', 'suffix-3': u'erm', 'has_hyphen': True, 'is_numeric': False, 'next_word': u'outperform', 'is_last': False, 'suffix-1': u'm', 'suffix-2': u'rm', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'lo', 'prev_word': u'a'}, {'is_first': False, 'word': u'outperform', 'is_all_lower': True, 'prefix-3': u'out', 'is_capitalized': False, 'prefix-1': u'o', 'suffix-3': u'orm', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'rating', 'is_last': False, 'suffix-1': u'm', 'suffix-2': u'rm', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'ou', 'prev_word': u'long-term'}, {'is_first': False, 'word': u'rating', 'is_all_lower': True, 'prefix-3': u'rat', 'is_capitalized': False, 'prefix-1': u'r', 'suffix-3': u'ing', 'has_hyphen': False, 'is_numeric': False, 'next_word': u'.', 'is_last': False, 'suffix-1': u'g', 'suffix-2': u'ng', 'is_all_caps': False, 'capitals_inside': False, 'prefix-2': u'ra', 'prev_word': u'outperform'}, {'is_first': False, 'word': u'.', 'is_all_lower': True, 'prefix-3': u'.', 'is_capitalized': True, 'prefix-1': u'.', 'suffix-3': u'.', 'has_hyphen': False, 'is_numeric': False, 'next_word': '', 'is_last': True, 'suffix-1': u'.', 'suffix-2': u'.', 'is_all_caps': True, 'capitals_inside': False, 'prefix-2': u'.', 'prev_word': u'rating'}]]

y_pred = crf.predict(X_test)
print(X_test)
print(metrics.flat_f1_score(y_test, y_pred,
					  average='weighted', labels=labels))




