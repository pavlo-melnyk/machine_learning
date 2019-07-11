import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 

from util import get_normalized_data, getData, getBinaryData, y2indicator, swish, error_rate, init_weight_and_bias
from sklearn.utils import shuffle
from datetime import datetime 



class HiddenLayer(object):
	def __init__(self, M1, M2, an_id):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2
		W, b = init_weight_and_bias(M1, M2)
		self.W = tf.Variable(W.astype(np.float32), name='W%s' % self.id)
		self.b = tf.Variable(b.astype(np.float32), name='b%s' % self.id)
		self.parameters = [self.W, self.b]


	def forward(self, X, activation=tf.nn.relu):
		return activation(tf.matmul(X, self.W) + self.b)



class ANN(object):
	def __init__(self, hidden_layer_sizes, p_keep):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.dropout_rates = p_keep

	def fit(self, X, Y, learning_rate=1e-4, mu=0.9, decay=0.9, epochs=15, batch_sz=100, display_cost=False, save_params=False):
		# set evarything to np.float32 to enable tf computation running correctly
		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)
			
		# create a vailidation set:		
		X, Y = shuffle(X, Y)

		Xvalid, Yvalid = X[-1000:,], Y[-1000:]		
		X, Y  = X[:-1000,], Y[:-1000]
		
		
		# initialize hidden layers:
		N, D = X.shape
		K = len(set(Y))
		self.hidden_layers = []
		M1 = D 
		count = 0
		# iterate the self.hidden_layer_sizes list through M1 variable:
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2, count)
			self.hidden_layers.append(h)
			M1 = M2
			count += 1

		# the last_hidden_layer-output_layer weights and bias:  
		W, b = init_weight_and_bias(M1, K)
		self.W = tf.Variable(W, name='W%s' % count)
		self.b = tf.Variable(b, name='b%s'% count)

		# collect all the network's parameters:
		self.params = [self.W, self.b]
		for h in self.hidden_layers:
			self.params += h.parameters

		# define tensorflow placeholders:
		tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
		tfT = tf.placeholder(tf.int32, shape=(None,), name='T')

		# the logits ouputs of the network:
		Y_logits = self.forward_train(tfX)

		# define the expression for cost:
		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y_logits, labels=tfT))
		
		# define the tensorflow train function:
		train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)
	
		predict_op = self.predict(tfX)

		# validation cost will be calculated separately since nothing will be dropped
		Y_logits_valid = self.forward_predict(tfX)
		cost_valid = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y_logits_valid, labels=tfT))

		n_batches = N // batch_sz
		costs = []
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			# initialize all tf variables:
			print('\nInitializing variables...')
			session.run(init)
			print('\nPerforming batch SGD with RMSProp and momentum...')
			for i in range(epochs):				
				X, Y = shuffle(X, Y)
				for j in range(n_batches):
					Xbatch = X[j*batch_sz:(j+1)*batch_sz, :]
					Ybatch = Y[j*batch_sz:(j+1)*batch_sz]

					session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})
					if j % 20 == 0:
						c = session.run(cost_valid, feed_dict={tfX: Xvalid, tfT: Yvalid})
						costs.append(c)
						prediction = session.run(predict_op, feed_dict={tfX: Xvalid, tfT: Yvalid})
						#print(prediction)
						error = error_rate(Yvalid, prediction)
						print('\ni: %d,  j: %d,  cost: %.6f,  error: %.6f' % (i, j, c, error))

			# make the final prediction:			
			prediction = session.run(predict_op, feed_dict={tfX: Xvalid})
			final_error = error_rate(Yvalid, prediction)

			if save_params:	
				for h in self.hidden_layers:
					p_type = 'W'
					for p in h.parameters:						
						p = p.eval()	
						#print(type(p))
						#print(p.shape)	
						name = p_type + str(h.id)
						np.save(name, p)
						p_type = 'b'
				# last hidden layer - output layer parameters:
				np.save('W%s' % count, self.W.eval())
				np.save('b%s'% count, self.b.eval())
			
		if display_cost:
			plt.plot(costs)
			plt.show()


	def forward_train(self, X):
		# tf.nn.dropout scales inputs by 1/p_keep
		# therefore, during test time, we don't have to scale again
		Z = X
		Z = tf.nn.dropout(Z, self.dropout_rates[0])
		for h, p in zip(self.hidden_layers, self.dropout_rates[1:]):
			Z = h.forward(Z)
			Z = tf.nn.dropout(Z, p)
		# the output layer:
		logits = tf.matmul(Z, self.W) + self.b
		return logits


	def predict(self, X):
		Y_logits = self.forward_predict(X)
		return tf.argmax(Y_logits, 1)


	def forward_predict(self, X):
		Z = X		
		
		# below is wrong:
		# Z = Z * self.dropout_rates[0]
		# for h, p in zip(self.hidden_layers, self.dropout_rates):
		# 	Z = h.forward(Z) * p

		# this one is correct - we don't need to scale during test time,
		# b/c during training, the weights are updated when neurons' outputs
		# are scaled by 1/p_keep:
		for h in self.hidden_layers:
			Z = h.forward(Z)

		# the output layer:
		logits = tf.matmul(Z, self.W) + self.b
		return logits



def main():
	# X, Y = getData()
	#X, Y = getBinaryData()
	X, Y = get_normalized_data()

	t0 = datetime.now()
	model = ANN([500, 300], [0.8, 0.8, 0.8])
	model.fit(X, Y, display_cost=True, save_params=False)
	dt = datetime.now() - t0

	print('Elapsed time:', dt)

			


if __name__ == '__main__':
	main()