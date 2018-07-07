import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 

from util import get_normalized_data, y2indicator, swish, error_rate, init_weight_and_bias
from sklearn.utils import shuffle
from datetime import datetime 



def init_weight(M1, M2):
	return np.random.randn(M1, M2) / np.sqrt(M1 / 2)


class HiddenLayerBatchNorm(object):
	def __init__(self, M1, M2, an_id):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2

		W = init_weight(M1, M2).astype(np.float32)
		gamma = np.ones(M2).astype(np.float32)
		betta = np.zeros(M2).astype(np.float32)


		self.W = tf.Variable(W, name='W%s' % self.id)
		self.gamma = tf.Variable(gamma, name='gamma%s' % self.id)
		self.betta = tf.Variable(betta, name='betta%s' % self.id)
		
		# for test step we are to need a running-mean and a running-variance:
		self.rn_mean = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)
		self.rn_var = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)


	def forward(self, X, is_training, decay=0.9):
		Z = tf.matmul(X, self.W)
		if is_training:
			batch_mean, batch_var = tf.nn.moments(Z, [0])
			# update the running mean and running variance:
			update_rn_mean = tf.assign(
				self.rn_mean,
				self.rn_mean * decay + batch_mean * (1 - decay)
			)

			update_rn_var = tf.assign(
				self.rn_var, 
				self.rn_var * decay + batch_var * (1 - decay)
			)
			# to make sure the aforementioned updates are calculated
			# every time we call the train function,
			# we have to use next function:
			with tf.control_dependencies([update_rn_mean, update_rn_var]):
				Z = tf.nn.batch_normalization(
					Z,
					batch_mean, 
					batch_var, 
					self.betta, 
					self.gamma, 
					1e-4
				)
		else: 
			Z = tf.nn.batch_normalization(
				Z, 
				self.rn_mean, 
				self.rn_var, 
				self.betta, 
				self.gamma, 
				1e-4
			)

		return swish(Z, tensorflow=True)
		#return tf.nn.relu(Z)


class HiddenLayer(object):
	def __init__(self, M1, M2, activation):
		self.activation = activation
		self.M1 = M1
		self.M2 = M2
		W, b = init_weight_and_bias(M1, M2)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))
		self.parameters = [self.W, self.b]


	def forward(self, X):
		return tf.matmul(X, self.W) + self.b



class ANN(object):
	def __init__(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes


	def set_session(self, session):
		self.session = session


	def fit(self, X, Y, learning_rate=1e-2, mu=0.99, decay=0.99, epochs=15, batch_sz=100, display_cost=False, save_params=False):
		# set evarything to np.float32 to enable tf computation running correctly
		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)
				
		# create a vailidation set:		
		X, Y = shuffle(X, Y)
		X, Y = X.astype(np.float32), Y
		Xvalid, Yvalid = X[-1000:,], Y[-1000:]
		X, Y  = X[:-1000,], Y[:-1000]
		
		# initialize hidden layers:
		N, D = X.shape
		K = len(set(Y))
		self.layers = []
		M1 = D 
		count = 0
		# iterate the self.hidden_layer_sizes list through M1 variable:
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayerBatchNorm(M1, M2, count)
			self.layers.append(h)
			M1 = M2
			count += 1

		# the output layer:  
		h = HiddenLayer(M1, K, lambda x: x)	
		self.layers.append(h)

		if batch_sz == None:
			batch_sz = N

		# define tensorflow placeholders:
		tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
		tfT = tf.placeholder(tf.int32, shape=(None,), name='T')

		# for later use:
		self.tfX = tfX

		# training step - the logits ouputs of the network:
		logits = self.forward(tfX, is_training=True)

		# define the expression for cost:
		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tfT))
		
		# define the tensorflow train function:
		#train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu, use_nesterov=True).minimize(cost)
		#train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)
		train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
		#train_op = tf.train.GradientDescentOptimizer(leatning_rate).minimize(cost)

		# recall: forward propagation for test (prediction) step is different from the train one;
		# so we need to calculate logits in a different way:
		valid_logits = self.forward(tfX, is_training=False)
		self.predict_op = tf.argmax(valid_logits, 1)
		cost_valid = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=valid_logits, labels=tfT))

		n_batches = N // batch_sz
		costs = []
		# initialize all tf variables:
		print('\nInitializing variables...')			
		init = tf.global_variables_initializer()
		self.session.run(init)
	
		print('\nPerforming batch SGD with Nesterov momentum...')
		for i in range(epochs):				
			X, Y = shuffle(X, Y)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j+1)*batch_sz, :]
				Ybatch = Y[j*batch_sz:(j+1)*batch_sz]

				self.session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})
				if j % 20 == 0:
					c, lgts = self.session.run([cost_valid, valid_logits], feed_dict={tfX: Xvalid, tfT: Yvalid})
					costs.append(c)
					prediction = self.predict(Xvalid)
					#print(Yvalid)
					#print(prediction)
					#print(np.argmax(lgts, 1))
					error = error_rate(Yvalid, prediction)
					print('\ni: %d,  j: %d,  cost: %.6f,  error: %.6f' % (i, j, c, error))
					print('dbg:', self.session.run(self.layers[0].rn_mean).sum())
		# make the final print:	
		print('Train acc:', self.score(X, Y), 'Test acc:', self.score(Xvalid, Yvalid))		
		
		if display_cost:
			plt.plot(costs)
			plt.show()


	def forward(self, X, is_training):
		Z = X
		for h in self.layers[:-1]:
			Z = h.forward(Z, is_training)
		# the output layer:
		Y_logits = self.layers[-1].forward(Z)
		return Y_logits


	def score(self, X, Y):
		P = self.predict(X)
		return np.mean(Y == P)


	def predict(self, X):
		return self.session.run(self.predict_op, feed_dict={self.tfX: X})



def main():
	X, Y = get_normalized_data()

	t0 = datetime.now()

	model = ANN([500, 300])

	session = tf.InteractiveSession()
	model.set_session(session)

	model.fit(X, Y, display_cost=True, save_params=True)

	dt = datetime.now() - t0

	print('Elapsed time:', dt)

			


if __name__ == '__main__':
	main()