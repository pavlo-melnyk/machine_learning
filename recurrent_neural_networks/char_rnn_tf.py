''' 
An API for character-level RNNs utilizing TensorFlow.
Inspired by Minimal character-level Vanilla RNN model written by Andrej Karpathy (@karpathy):
https://gist.github.com/karpathy/d4dee566867f8291f086 .
Learns from the input txt file.
Supportst multilayer RNNs.
Generates text during the training process and writes it to a file.
To be improved...
'''

import sys
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import brown

from tensorflow.contrib.rnn import BasicRNNCell as Basic, GRUCell as GRU, LSTMCell as LSTM
from util import init_weight, get_char_rnn_data
from datetime import datetime


class CharRNN:
	def __init__(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes
		
	def fit(self, X, T, encoding='gbk', learning_rate=1e-1,  mu=0.9, epochs=10, display_cost=True, activation=tf.nn.relu, RecurrentUnit=GRU):
		# preprocess the data:
		chars = list(set(X))
		self.N, self.V = len(X), len(chars)
		self.T = T

		char2ix = { ch:i for i, ch in enumerate(chars) }
		ix2char = { i:ch for i, ch in enumerate(chars) }

		# for convenience:
		N = self.N
		V = self.V
		hidden_layer_sizes = self.hidden_layer_sizes
		self.f = activation

		print('\ndata has %d characters, %d unique.\n' % (N, V))

		# create and initialize
		hidden_layers = []
		# create hidden layers - recurrent units:
		for M in hidden_layer_sizes:
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
		tfY = tf.placeholder(tf.int32, shape=(None, ), name='targets')	
	
		# create placeholders for sequence_length
		# (needed in the tf.nn.dynamic_rnn function):
		self.tfT = tf.placeholder(tf.int32, shape=(), name='T')

		# forward-propagate the data:
		Z, states = tf.nn.dynamic_rnn(
			cell=hidden_layers, 
			inputs=tf.reshape(self.tfX, (1, self.tfT, V)), 
			dtype=tf.float32
		)

		self.logits = tf.matmul(Z[0], self.Wo) + self.bo
		self.py_x = tf.nn.softmax(self.logits)

		# define the operations:
		self.predict_op = tf.argmax(self.logits, 1)

		cost_op = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=self.logits,
				labels=tfY
			)
		)

		# train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)
		train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)


		# training:
		costs = []		
		init = tf.global_variables_initializer()
		n_sequences = N // T + 1
		
		with tf.Session() as session, open('generated_text.txt', 'w', encoding=encoding) as f:
			self.session = session
			session.run(init)
			print()
			for i in range(epochs):
				t0 = datetime.now()
				accs = [] # storage for accuracies
				cost = -np.log(1.0 / V) * T # initial cost value
				for j in range(n_sequences):
					inputs = [char2ix[char] for char in X[j:j+T]]
					targets = [char2ix[char] for char in X[j+1:j+T+1]]

					# encode inputs in 1-of-k representation (make it one-hot encoded):
					xs = np.zeros((T, V))
					for t in range(T):
						xs[t, inputs[t]] = 1
					# xs[np.arange(T), inputs] = 1 # same using numpy array indexing
					xs = np.expand_dims(xs, axis=0) # needs to be size (batch_sz, T, V)
													# for passing in tf.nn.dynamic_rnn
					# print('\n', xs.shape)
					# print(len(targets))

					_, c, predictions = session.run([train_op, cost_op, self.predict_op], feed_dict={self.tfX: xs, self.tfT: T, tfY: targets})
					cost =  cost * 0.999 + c * 0.001
					if j % 100 == 0:
						acc = np.sum(targets==predictions) / T
						# print('targets.shape:', len(targets), ' predictions.shape:', len(predictions))
						# exit()
						accs.append(acc) # use the average accuracy among all sequences
						sys.stdout.write(
							'---> epoch: %d/%d, j/n_sequences: %d/%d, smooth loss: %0.4f, acc so far: %0.4f\r' \
							 % (i+1, epochs, j+1, n_sequences, cost, np.sum(accs)/len(accs))
						)
						sys.stdout.flush()
						
					if j % 500 == 0:
						# sample from the model:
						# print('----> generating and writing to a file\n')
						sample_ix = self.sample(np.random.choice(V), 200)
						txt = ''.join(ix2char[ix] for ix in sample_ix)
						# print('\n-----\n %s \n-----\n\n' % (txt, ))
						f.write('epoch: %d, j: %d\n-----\n%s\n-----\n\n' % (i+1, j+1, txt, ))

				costs.append(cost) # add the final 'smooth' cost value to display layter
				print('---> epoch: %d/%d, time elapsed: %s, loss: %f, acc: %0.6f           ' \
				 % (i+1, epochs, datetime.now()-t0, cost, np.sum(accs)/len(accs)))
								
			if display_cost:
				plt.plot(costs)
				plt.ylabel('cost')
				plt.xlabel('epochs')
				plt.show()
				

	def sample(self, seed_ix, n):
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
			py_x = self.session.run(self.py_x, feed_dict={self.tfX: np.expand_dims(x, axis=0), self.tfT: t+1})
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


def train_and_generate(filename='kobzar.txt', encoding='utf-8'):
	# load the data:
	X = get_char_rnn_data(filename=filename, encoding=encoding)
	# X = brown.get_sentences(return_str=True)

	# create a RNN and fit the data:
	rnn = CharRNN([100])
	rnn.fit(X, T=100, encoding=encoding, epochs=11, learning_rate=1e-3, RecurrentUnit=Basic)


if __name__ == '__main__':
	train_and_generate()