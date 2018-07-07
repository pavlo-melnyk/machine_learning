import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 

from util import get_normalized_data, swish, error_rate, init_weight_and_bias
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


	def forward(self, X):
		return swish(tf.matmul(X, self.W) + self.b, tensorflow=True)



class ANN(object):
	def __init__(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes


	def fit(self, X, Y, learning_rate=1e-4, mu=0.9, decay=0.99, reg=1e-3, epochs=10, batch_sz=100, display_cost=False, save_params=False):
		# set evarything to np.float32 to enable tf computation running correctly
		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)
		reg = np.float32(reg)
		
		# create a vailidation set:		
		X, Y = shuffle(X, Y)
		Xvalid, Yvalid = X[-1000:,], Y[-1000:]
		Yvalid_ind = y2indicator(Yvalid)
		X, Y  = X[:-1000,], Y[:-1000]
		Y_ind = y2indicator(Y)
		
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
		tfT = tf.placeholder(tf.float32, shape=(None, K), name='T')

		# the logits ouputs of the network:
		Y_logits = self.forward(tfX)

		# define the expression for cost:
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=tfT))
		regularization = reg * sum([tf.nn.l2_loss(p) for p in self.params])
		cost += regularization

		# define the tensorflow train function:
		train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)
		#train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
		predict_op = self.predict(tfX)

		n_batches = N // batch_sz
		costs = []
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			# initialize all tf variables:
			print('\nInitializing variables...')
			session.run(init)
			print('\nPerforming batch SGD with RMSProp and momentum...')
			for i in range(epochs):				
				X, Y, Y_ind = shuffle(X, Y, Y_ind)
				for j in range(n_batches):
					Xbatch = X[j*batch_sz:(j+1)*batch_sz, :]
					Ybatch = Y_ind[j*batch_sz:(j+1)*batch_sz, :]

					session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})
					if j % 20 == 0:
						c = session.run(cost, feed_dict={tfX: Xvalid, tfT: Yvalid_ind})
						costs.append(c)
						prediction = session.run(predict_op, feed_dict={tfX: Xvalid, tfT: Yvalid_ind})
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


	def forward(self, X):
		Z = X
		for h in self.hidden_layers:
			Z = h.forward(Z)
		# the output layer:
		Y_logits = tf.matmul(Z, self.W) + self.b
		return Y_logits


	def predict(self, X):
		Y_logits = self.forward(X)
		return tf.argmax(Y_logits, 1)



def main():
	X, Y = get_normalized_data()

	t0 = datetime.now()
	model = ANN([2000, 1000, 500])
	model.fit(X, Y, display_cost=True, save_params=True)
	dt = datetime.now() - t0

	print('Elapsed time:', dt)

			


if __name__ == '__main__':
	main()