import numpy as np 
import string 
import os
import sys
import operator
import codecs

from nltk import pos_tag, word_tokenize
from datetime import datetime


def init_weight(Mi, Mo):
	return (np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)).astype(np.float32)


def get_char_rnn_data(filename='kobzar.txt', encoding=None, return_vocab_size=False):
	# input and preprocess the data:
	data = codecs.open(filename, 'r', encoding=encoding).read()
	vocab_size = len(set(data))
	if return_vocab_size:
		return data, vocab_size
	return data