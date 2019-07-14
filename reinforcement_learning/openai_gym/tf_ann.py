import os
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 



class HiddenLayer:
	def __init__(self, M1, M2, an_id):
		self.an_id = an_id
		w_init = np.random.randn(M1, M2) / np.sqrt(M1/2) # He-normal initialization
		b_init = np.zeros(M2)		
		self.w = tf.Variable(w_init.astype(np.float32), name='W_' + str(an_id))
		self.b = tf.Variable(b_init.astype(np.float32), name='b_' + str(an_id))

	def forward(self, X, activation):
		return activation( tf.matmul(X, self.w,) + self.b )


	
class ANN:
	def __init__(self, D, hidden_layer_sizes, dropout_rates=None, activation=tf.nn.relu, lr=1e-1, l2=1e-1):
		print('\nHello, TensorFlow!')
		if not dropout_rates:
			self.forward_train = self.forward
		else:
			# p_keep for input and hidden layers:
			assert len(dropout_rates) == len(hidden_layer_sizes) + 1
			self.dropout_rates = dropout_rates
			
		# hidden layers activation function:
		self.activation = activation 

		# the learning rate and l2 regularization parameter:
		self.lr = lr
		self.l2 = l2
		
		# create placeholders for input data:
		self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
		self.T = tf.placeholder(tf.float32, shape=(None,), name='T') # labels

		# create hidden layers:
		self.layers = []
		M1 = D
		for i, M2 in enumerate(hidden_layer_sizes):
			self.layers.append( HiddenLayer(M1, M2, i+1) )
			M1 = M2

		# and the output layer:
		self.layers.append( HiddenLayer(M1, 1, 'output') )

		# model output:
		self.Y = tf.squeeze( self.forward_train(self.X), axis=1 ) # 1-D

		# model output for inference time:
		self.Ytest = tf.squeeze( self.forward(self.X), axis=1 )

		# squared error:
		self.cost = tf.reduce_sum(tf.square(self.Y - self.T))
		self.cost += l2 * tf.add_n([ tf.reduce_sum(tf.square(v))/2 for v in tf.trainable_variables()])
		# cost += l2 * tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()])

		# training function:
		self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)

		# define an interactive session, s.t. it could be used within other functions:
		self.sess = tf.InteractiveSession()
		
		# initialize tf variables: 
		init = tf.global_variables_initializer()
		self.sess.run(init)


	def forward_train(self, X):
		print('\nperforming dropout', end='\n\n')
		# dropout on the input layer:
		Z = tf.nn.dropout(X, self.dropout_rates[0])
		for p_keep, layer in zip(self.dropout_rates[1:], self.layers[:-1]):
			Z = tf.nn.dropout(layer.forward(Z, self.activation), p_keep)
		return self.layers[-1].forward(Z, tf.identity)


	def forward(self, X):
		Z = X
		for layer in self.layers[:-1]:
			Z = layer.forward(Z, self.activation)
		return self.layers[-1].forward(Z, tf.identity)


	def partial_fit(self, X, T):
		_, cost = self.sess.run([self.train_op, self.cost], feed_dict={self.X: X, self.T: T})
		return cost


	def predict(self, X):
		return self.sess.run(self.Ytest, feed_dict={self.X: X})


	def get_r_sq(self, X, T):
		d1 = T - self.predict(X)
		d2 = T - T.mean()
		return 1 - d1.dot(d1) / d2.dot(d2)



def normalize(X):
	return (X - X.mean(axis=0)) / X.std(axis=0)



def main():
	# load data:
	from keras.datasets import boston_housing
	(Xtrain, Ttrain), (Xtest, Ttest) = boston_housing.load_data()

	# normalize_data:
	Xtrain, Xtest = normalize(Xtrain), normalize(Xtest)
	print('\nXtrain.shape, Ttrain.shape:', Xtrain.shape, Ttrain.shape)
	print('Xtest.shape, Ttest.shape:', Xtest.shape, Ttest.shape, end='\n\n')

	# train the model:
	# dropout_rates = [0.9, 0.7, 0.7, 0.7]
	dropout_rates = None
	model = ANN(Xtrain.shape[1], [100, 100, 100], dropout_rates, lr=1e-5, l2=0)
	
	for i in range(2000):
		model.partial_fit(Xtrain, Ttrain)

	print('\ntrain r_squared:', model.get_r_sq(Xtrain, Ttrain))
	print('test r_squared:', model.get_r_sq(Xtest, Ttest))
	print('\nTtrain[:10]:', Ttrain[:10])
	print('\nmodel.predict(Xtrain[:10]):', model.predict(np.array(Xtrain[:10])))
	


if __name__ == '__main__':
	main()