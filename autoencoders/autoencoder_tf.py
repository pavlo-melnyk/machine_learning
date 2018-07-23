import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from utils import get_mnist_data, init_weights_and_biases



class HiddenLayer:
	def __init__(self, M1, M2, an_id):
		W0, b0 = init_weights_and_biases(M1, M2)
		self.W = tf.Variable(W0, name='W%s'%an_id)
		self.b = tf.Variable(b0, name='b%s'%an_id)
		self.params = [self.W, self.b]

	def forward(self, X):
		return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class Autoencoder:
	def __init__(self, D, hidden_layer_sizes, loss_fn='cross-entropy'):
		self.hidden_layer_sizes = hidden_layer_sizes

		# input batch of training data (batch_size x D):
		self.X = tf.placeholder(tf.float32, shape=(None, D))

		# create hidden layers:
		self.hidden_layers = []
		M1 = D

		for i, M2 in enumerate(self.hidden_layer_sizes):
			h = HiddenLayer(M1, M2, i)
			self.hidden_layers.append(h)
			M1 = M2
			
		# hidden --> output layer parameters:
		Wo, bo = init_weights_and_biases(M2, D)
		self.Wo = tf.Variable(Wo, name='Wo') # (M2 x D)
		self.bo = tf.Variable(bo, name='bo') # D

		# collect all network parameters:
		self.params = []
		for h in self.hidden_layers:
			self.params += h.params
		self.params += [self.Wo, self.bo]
		# print('self.params:', self.params)

		# get output - our reconstruction:
		logits = self.forward(self.X)

		if loss_fn == 'cross-entropy':
			# assuming inputs and outputs to be Bernoulli probabilities
			self.output = tf.nn.sigmoid(logits)
			# define the cost function:
			self.cost = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(
					labels=self.X, 
					logits=logits
				)
			)

		elif loss_fn == 'mse':
			# assuming the difference (error) between inputs and outputs to be Gaussian
			self.output = tf.nn.sigmoid(logits)	# assuming output is in range [0, 1]	
			# self.output = logits
			self.cost = tf.reduce_mean(
				tf.losses.mean_squared_error(
					labels=self.X,
					predictions=self.output
				)
			)

		# define session:		
		self.sess = tf.InteractiveSession()
		


	def fit(self, X, epochs=30, learning_rate=0.001, mu=0.9, decay=0.9, batch_size=64):
		train_op = tf.train.RMSPropOptimizer(
			learning_rate=learning_rate,
			momentum=mu,
			decay=decay
		).minimize(self.cost)

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		costs = []
		n_batches = len(X) // batch_size
		print('n_batches:', n_batches)
		for i in range(epochs):
			print('epoch:', i)
			np.random.shuffle(X)
			for j in range(n_batches): 	
				Xbatch = X[j*batch_size:(j+1)*batch_size]
				_, c = self.sess.run((train_op, self.cost), feed_dict={self.X: Xbatch})
				costs.append(c)
				if j % 100 == 0:
					print('j: %d, cost: %.3f' % (j, c))

		plt.plot(costs)
		plt.xlabel('epochs')
		plt.ylabel('cost')
		plt.show()


	def forward(self, X):
		# forward-pass through all the hidden layers:
		Z = X		
		for h in self.hidden_layers:
			Z = h.forward(Z)			

		# return logits:
		return tf.matmul(Z, self.Wo) + self.bo


	def predict(self, X):
		return self.sess.run(self.output, feed_dict={self.X: X})



def main():
	X, Y = get_mnist_data(normalize=True) # images' pixels will be in range [0, 1]
	N, D = X.shape
	Xtest, Ytest = X[-100:], Y[-100:]
	X, Y = X[:-100], Y[:-100]

	model = Autoencoder(D, [300], loss_fn='cross-entropy')
	model.fit(X, epochs=15)

	# display reconstruction: 
	while True:
		i = np.random.choice(len(Xtest))
		x = Xtest[i]
		image = model.predict(np.expand_dims(x, axis=0)).reshape(28, 28) # or just use [x]: we need to input a matrix, not a vector
		plt.subplot(1, 2, 1)
		plt.imshow(x.reshape(28, 28), cmap='gray')
		plt.title('Original Image')
		plt.subplot(1, 2, 2)
		plt.imshow(image, cmap='gray')
		plt.title('Reconstruction')
		plt.show()

		prompt = input('Continue generating?\n')
		if prompt and prompt[0] in ['n', 'N']:
			break


if __name__ == '__main__':
	main()
