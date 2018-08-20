import numpy as np
import pandas as pd
import os
class Lang:
	def __init__(self, name):
		self.SOS_token = 0
		self.EOS_token = 1
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {}
		self.n_words = 0  # Count SOS and EOS
	def addSentence(self, sentence):
		for word in sentence.split(' '):
		    self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
		    self.word2index[word] = self.n_words
		    self.word2count[word] = 1
		    self.index2word[self.n_words] = word
		    self.n_words += 1
		else:
		    self.word2count[word] += 1

def buildDict(dataset):
	lang_word = Lang("word")
	lang_char = Lang("char")
	all_chars = []
	all_words = [] 
	for data in dataset:	
		for line in data:
			for sen in line.split('\n'):
				for word in sen.split(' '):
					all_words.append(word)
				[all_chars.append(char) for char in sen]
		
	[lang_char.addWord(char)for char in sorted(all_chars)]
	[lang_word.addWord(word)for word in sorted(all_words)]
	
	return lang_word, lang_char        
def sentence2index(lang, lines):
	index = []
	if   lang.name=='word':
		for sen in lines.split('\n'):
			for word in sen.split(' '):
				index.append(lang.word2index[word])
	elif lang.name=='char':
		for char in sen:
			index.append(lang.word2index[char])
	#index.append(1)
	return np.array(index)

def index2sentence(lang, index):
	sen = [lang.index2word[idx] for idx in index]
	if lang.name=='char':
		return "".join(sen[1:-1])
	elif lang.name=='word':
		return " ".join(sen[1:-1])
def filter_sen(word):
	return word.upper().replace('X', '')

if __name__=='__main__':

	data_dir  = 'data/images'
	name      = 'train'
	train_table     = pd.read_pickle(os.path.join(data_dir,'{}_table.pkl'.format(name)))	
	train_reports   = train_table.report
	train_reports   = [filter_sen(report) for report in train_reports]
	name          = 'val'
	val_table     = pd.read_pickle(os.path.join(data_dir,'{}_table.pkl'.format(name)))	
	val_reports   = val_table.report
	val_reports   = [filter_sen(report)for report in val_reports]
	
	
	lang_word, lang_char = buildDict([val_reports, train_reports])
	#print(lang_char.word2index)
	print(lang_word.word2index)
	
