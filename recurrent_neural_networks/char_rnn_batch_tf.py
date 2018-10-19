''' 
An API for character-level RNNs utilizing TensorFlow.
Inspired by Minimal character-level Vanilla RNN model written by Andrej Karpathy (@karpathy):
https://gist.github.com/karpathy/d4dee566867f8291f086 .
Learns from the input txt file.
Supports multilayer RNNs, saving, and batch training.
Generates text during the training process and writes it to a file.
'''

import sys
import json
import tensorflow as tf

import numpy as np 
import matplotlib.pyplot as plt 
import brown

from tensorflow.contrib.rnn import BasicRNNCell as Basic, GRUCell as GRU, LSTMCell as LSTM

from util import init_weight, get_char_rnn_data
from datetime import datetime


class CharRNN:
	str2unit = {'Basic': Basic, 'GRU': GRU, 'LSTM': LSTM}
	unit2str = {v: k for k, v in str2unit.items()}

	def __init__(self, savefile, V=None, hidden_layer_sizes=None, activation=tf.nn.relu, RecurrentUnit='Basic', vocab=None):
		self.savefile = savefile
		if V and hidden_layer_sizes and activation and RecurrentUnit:
			RecurrentUnit = CharRNN.str2unit[RecurrentUnit] # define the recurrent unit
			self.build(V, hidden_layer_sizes, activation, RecurrentUnit)
			self.ix2char = vocab
			
	def build(self, V, hidden_layer_sizes, activation, RecurrentUnit):
		tf.reset_default_graph()
		# save for later:
		self.V = V
		self.hidden_layer_sizes = hidden_layer_sizes 
		self.f = activation
		self.RecurrentUnit = RecurrentUnit

		# create and initialize
		hidden_layers = []
		# create hidden layers - recurrent units:
		for M in self.hidden_layer_sizes:
			hidden_layers.append(RecurrentUnit(num_units=M, activation=self.f))
		hidden_layers = tf.nn.rnn_cell.MultiRNNCell(hidden_layers)

		# initialize logistic-regression layer:
		Wo = init_weight(M, V).astype(np.float32)
		bo = np.zeros(V).astype(np.float32)

		# Wx, Wh, bh, etc. will be created by RecurrentUnit

		self.Wo = tf.Variable(Wo)
		self.bo = tf.Variable(bo)

		# collect all the params:
		self.params = [p for p in tf.trainable_variables()]
		
		# create placeholders for input and output:		
		self.tfX = tf.placeholder(tf.float32, shape=(None, None, V), name='inputs')
		self.tfY = tf.placeholder(tf.int32, shape=(None, ), name='targets')	
	
		# create placeholders for sequence_length and batch_sz
		# (needed in the tf.nn.dynamic_rnn function):
		self.tfT = tf.placeholder(tf.int32, shape=(), name='T')
		self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')

		# forward-propagate the data:
		Z, states = tf.nn.dynamic_rnn(
			cell=hidden_layers, 
			inputs=self.tfX, # tf.reshape(self.tfX, (self.batch_size, self.tfT, V)),
			dtype=tf.float32,
			parallel_iterations=256,
			swap_memory=True
		)
		# print('type(Z):', type(Z))
		# print('Z.shape:', Z.shape)
		# exit()
		Z = tf.reshape(Z, (self.batch_size*self.tfT, Z.shape[2])) # (batch_size*T x last_hidden_layer_size)

		self.logits = tf.matmul(Z, self.Wo) + self.bo # batch_size*T x V
		self.py_x = tf.nn.softmax(self.logits)

		self.saver = tf.train.Saver()

		# define the operations:
		self.predict_op = tf.argmax(self.logits, 1)

		self.cost_op = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=self.logits,
				labels=self.tfY
			)
		)

		
	def fit(self, X, T, encoding='gbk', learning_rate=1e-1,  mu=0.9, l2 = 0.01, epochs=10, batch_sz=32, display_cost=True):
		# preprocess the data:
		chars = list(set(X))
		N, V = len(X), len(chars)		

		print('\ndata has %d characters, %d unique.\n' % (N, V))

		char2ix = { ch:i for i, ch in enumerate(chars) }
		ix2char = { i:ch for i, ch in enumerate(chars) }		

		# save for later:
		self.ix2char = ix2char

		# add l2-regularization to the cost:
		l2_reg = l2 * tf.reduce_mean([tf.reduce_mean(p)**2 for p in self.params]) / 2
		reg_cost = self.cost_op + l2_reg

		# define an optimizer:
		# train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(reg_cost)
		train_op = tf.train.AdamOptimizer(learning_rate).minimize(reg_cost)

		# training:
		costs = []		
		init = tf.global_variables_initializer()
		n_batches = N // (batch_sz*T)
				
		with tf.Session() as session, open('generated_text_batch.txt', 'w', encoding=encoding) as f:			
			session.run(init) 
			print()
			for i in range(epochs):
				t0 = datetime.now()
				accs = [] # storage for accuracies
				cost = -np.log(1.0 / V) * batch_sz*T # initial cost value - sum of expected values of predictions
				for j in range(n_batches):
					# create input sequence and target sequence
					# input X should be a 3-D array
					# first make lists of inputs and targets:					
					inputs = [char2ix[char] for char in X[j*batch_sz*T : (j+1)*batch_sz*T]] 
					targets = [char2ix[char] for char in X[j*batch_sz*T+1 : (j+1)*batch_sz*T+1]]
					# print('len(inputs):', len(inputs))
					# print('10 first inputs:', inputs[:10])
					# print('10 first targets:', targets[:10])
					# exit()

					# encode inputs in 1-of-k representation (make it one-hot encoded):
					xs = np.zeros((batch_sz*T, V)) # 2-D array of batch_size*T lenght
					for t in range(batch_sz*T):
						try:
							xs[t, inputs[t]] = 1
						except Exception as e:
							print('j:', j, 't:', t)
							print('len(targets):', len(targets))
							print(e)
							exit()
					# xs[np.arange(batch_sz*T), inputs] = 1 # same using numpy array indexing
					xs = np.reshape(xs, (batch_sz, T, V)) # must be size (batch_sz, T, V)
													      # for passing in tf.nn.dynamic_rnn

					# NOTE: we don't need to reshape the targets! leave them a vector of size (batch_sz*T,)
									
					# print('\ninput shape:', xs.shape)
					# print('targets shape:', len(targets))
					# exit()

					_, c, predictions = session.run(
											[train_op, reg_cost, self.predict_op], 
											feed_dict={self.tfX: xs, self.batch_size: batch_sz, self.tfT: T, self.tfY: targets}
										)

					cost =  cost * 0.999 + c * 0.001
					if j % (n_batches // 20) == 0:
						acc = np.sum(targets==predictions) / (batch_sz*T)
						# print('targets.shape:', len(targets), ' predictions.shape:', len(predictions))
						# exit()
						accs.append(acc) # use the average accuracy among all sequences
						sys.stdout.write(
							'---> epoch: %d/%d, j/n_batches: %d/%d, smooth loss: %0.4f, acc so far: %0.4f\r' \
							 % (i+1, epochs, j+1, n_batches, cost, np.sum(accs)/len(accs))
						)
						sys.stdout.flush()
						
					if j % (n_batches // 5) == 0:
						# sample from the model:
						# print('----> generating and writing to a file\n')
						sample_ix = self.sample(np.random.choice(V), 200, session)
						txt = ''.join(ix2char[ix] for ix in sample_ix)
						# print('\n-----\n %s \n-----\n\n' % (txt, ))
						f.write('epoch: %d, j: %d\n-----\n%s\n-----\n\n' % (i+1, j+1, txt, ))

				costs.append(cost) # add the final 'smooth' cost value to display layter
				print('---> epoch: %d/%d, time elapsed: %s, loss: %f, acc: %0.6f           ' \
				 % (i+1, epochs, datetime.now()-t0, cost, np.sum(accs)/len(accs)))
			
				self.saver.save(session, self.savefile)
			# print(self.params[0].eval())

			if display_cost:
				plt.plot(costs)
				plt.ylabel('cost')
				plt.xlabel('epochs')
				plt.show()
				
	def sample(self, seed_ix, n, session):
		'''
		Sample a sequence of integers from the model.
		seed_ix is seed letter for first time step.
		'''
		# get the first word:
		x = np.zeros((1, self.V))
		x[:, seed_ix] = 1
		ixes = [] # sampled sequence
		for t in range(n):
			# get the output for the sequence
			# (returns a softmax vector for every time step):
			py_x = session.run(self.py_x, feed_dict={self.tfX: np.expand_dims(x, axis=0), self.batch_size: 1, self.tfT: t+1})
			# print('py_x.shape:', py_x.shape) # (t+1, 124)
			# py_x = np.exp(y[-1]) / np.sum(np.exp(y[-1]))

			# we need the softmax output for the final time step
			# to sample from (py_x[-1].shape = (124, 1)):
			ix = np.random.choice(range(self.V), p=py_x[-1])

			# get the sampled word - prediction:
			x_next = np.zeros((1, self.V))
			x_next[:, ix] = 1

			# 'append' our sequence to pass it whole
			# at next time step:
			x = np.vstack((x, x_next)) # (t+1, V)

			# collect all the predicted (sampled) indeces:
			ixes.append(ix)
		return ixes

	def generate(self, n):	
		with tf.Session() as session:
			# restore the model
			self.saver.restore(session, self.savefile)
			for i in range(1):
				sample_ix = self.sample(np.random.choice(self.V), n, session)
				txt = ''.join(self.ix2char[str(ix)] for ix in sample_ix)
				print('\n-----\n %s \n-----\n\n' % (txt, ))
			# print(self.params[0].eval())

	def save(self, filename):
		# TODO: ideally, we have to save the activation and RecurrentUnit
		j = {			
			'V': self.V,
			'hidden_layer_sizes': self.hidden_layer_sizes,
			'model': self.savefile,
			'RecurrentUnit': CharRNN.unit2str[self.RecurrentUnit],
			'vocab': self.ix2char # must save to decode generated sequences later
		}
		with open(filename, 'w') as f:
			json.dump(j, f)

	@staticmethod
	def load(filename):
		with open(filename) as f:
			j = json.load(f)
		return CharRNN(
					j['model'], 
					j['V'], 
					j['hidden_layer_sizes'], 
					RecurrentUnit=j['RecurrentUnit'], 
					vocab=j['vocab']
				)
	

def train(filename='kobzar.txt', encoding='utf-8'):
	# get the data and vocab_size:
	X, V = get_char_rnn_data(filename=filename, encoding=encoding, return_vocab_size=True)
	# X = brown.get_sentences(return_str=True)
	rnn = CharRNN('./tf.model', V, [100], RecurrentUnit='Basic')
	rnn.fit(X, T=100, encoding=encoding, epochs=100, batch_sz=128, learning_rate=1e-3, l2=0.01)
	
	rnn.save('my_trained_model.json')
	

def generate(file='my_trained_model.json'):
	rnn = CharRNN.load(file)
	rnn.generate(200)

if __name__ == '__main__':
	train()
	generate()
	
