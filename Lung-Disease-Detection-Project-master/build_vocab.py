# adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
import pickle
import json
import os
import argparse
import pandas as pd
from collections import Counter



class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not (word in self.word2idx):
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return self.idx

        # def __getitem__(self, idx):



def build_vocab(captions, threshold=0):
	"""Build a simple vocabulary wrapper."""

	counter = Counter()
	for caption in captions:
		tokens  = caption.lower().split('/') 
		counter.update(tokens)


	# If the word frequency is less than 'threshold', then the word is discarded.
	words = [word for word, cnt in counter.items() if cnt >= threshold]
	# Creates a vocab wrapper and add some special tokens.
	vocab = Vocabulary()
	vocab.add_word('<pad>')
	vocab.add_word('<start>')
	vocab.add_word('<end>')
	vocab.add_word('<unk>')

	# Adds the words to the vocabulary.
	for i, word in enumerate(words):
		vocab.add_word(word)
	return vocab


def main(args):
    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--caption_path', type=str,
		                default='./data/mscoco/annotations/captions_train2014.json',
		                help='path for train annotation file')
	parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
		                help='path for saving vocabulary wrapper')
	parser.add_argument('--threshold', type=int, default=5,
		                help='minimum word count threshold')
	args = parser.parse_args()
	main(args)
	'''
	name = 'train'
	train_table      = pd.read_pickle(os.path.join('.','{}_table.pkl'.format(name)))	
	train_captions   = train_table.caption
	captions   = [caption for caption in train_captions]
	name          = 'val'
	val_table     = pd.read_pickle(os.path.join('.','{}_table.pkl'.format(name)))	
	val_captions   = val_table.caption
	val_captions   = [caption for caption in val_captions]
	captions.extend(val_captions)
	#print(train_captions)
	vocab = build_vocab(captions, threshold=1)

	with open('openi-vocab.pkl', 'wb') as f:
		pickle.dump(vocab, f)
	print("Total vocabulary size: %d" %len(vocab))

	
	
	
	
	
	
	
	
	
	
	
