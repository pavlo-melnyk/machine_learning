import numpy as np 
import theano
import theano.tensor as T
import matplotlib.pyplot as plt 

from utils import get_mnist_data, init_weights_and_biases


def rmsprop(cost, params, lr, mu, decay, eps):
	grads = T.grad(cost, params)
	updates = []
	for p, g in zip(params, grads):
		# cache
		ones = np.ones_like(p.get_value(), dtype=np.float32)
		c = theano.shared(ones)
		new_c = decay*c + (np.float32(1.0) - decay)*g*g

		# momentum:
		zeros = np.zeros_like(p.get_value(), dtype=np.float32)
		v = theano.shared(zeros)
		new_v = mu*v - lr*g / T.sqrt(new_c + eps)

		# param update:
		new_p = p + new_v

		# append the updates:
		updates.append((c, new_c))
		updates.append((v, new_v))
		updates.append((p, new_p))

	return updates


class HiddenLayer:
	def __init__(self, M1, M2, an_id):
		W0, b0 = init_weights_and_biases(M1, M2)
		self.W = theano.shared(W0, name='W%s'%an_id)
		self.b = theano.shared(b0, name='b%s'%an_id)
		self.params = [self.W, self.b]

	def forward(self, X):
		return T.nnet.relu(X.dot(self.W) + self.b)


class Autoencoder:
	def __init__(self, D, hidden_layer_sizes, loss_fn='sigmoid_cross-entropy'):
		self.hidden_layer_sizes = hidden_layer_sizes

		# input batch of training data (batch_size x D):
		self.X = T.matrix('X')

		# create hidden layers:
		self.hidden_layers = []
		M1 = D

		for i, M2 in enumerate(self.hidden_layer_sizes):
			h = HiddenLayer(M1, M2, i)
			self.hidden_layers.append(h)
			M1 = M2
			
		# hidden --> output layer parameters:
		Wo, bo = init_weights_and_biases(M2, D)
		self.Wo = theano.shared(Wo, name='Wo') # (M2 x D)
		self.bo = theano.shared(bo, name='bo') # D

		# collect all network parameters:
		self.params = []
		for h in self.hidden_layers:
			self.params += h.params
		self.params += [self.Wo, self.bo]
		# print('self.params:', self.params)

		# get output - our reconstruction:
		self.output = self.forward(self.X)		

		if loss_fn == 'sigmoid_cross-entropy':
			# assuming inputs and outputs to be Bernoulli probabilities
			# define the cost function:
			self.output = T.nnet.sigmoid(self.output)
			# self.cost = -T.mean(self.X*T.log(self.output) + (1 - self.X)*T.log(1 - self.output))
			self.cost = T.mean(
				T.nnet.binary_crossentropy(
					output=self.output,
					target=self.X
				)
			)

		elif loss_fn == 'mse':
			# assuming the difference (error) between inputs and outputs to be Gaussian
			self.output = T.nnet.sigmoid(self.output) # assuming output is in range [0, 1]				
			self.cost = T.mean((self.X - self.output)**2)

		self.predict = theano.function(
			inputs=[self.X],
			outputs=self.output
		)

		

	def fit(self, X, epochs=30, learning_rate=0.001, mu=0.9, decay=0.9, eps=1e-10, batch_size=64):
		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)
		eps = np.float32(eps)

		updates = rmsprop(self.cost, self.params, learning_rate, mu, decay, eps)

		train_op = theano.function(
			inputs=[self.X],
			updates=updates,
			outputs=self.cost
		)
		
		costs = []
		n_batches = len(X) // batch_size
		print('n_batches:', n_batches)
		for i in range(epochs):
			print('epoch:', i)
			np.random.shuffle(X)
			for j in range(n_batches): 	
				Xbatch = X[j*batch_size:(j+1)*batch_size]
				c = train_op(Xbatch)
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
		return Z.dot(self.Wo) + self.bo



def main():
	X, Y = get_mnist_data(normalize=True)
	N, D = X.shape
	Xtest, Ytest = X[-100:], Y[-100:]
	X, Y = X[:-100], Y[:-100]

	model = Autoencoder(D, [300], loss_fn='sigmoid_cross-entropy')
	model.fit(X, epochs=15)

	# display reconstruction: 
	while True:
		i = np.random.choice(len(Xtest))
		x = Xtest[i]
		image = model.predict(np.expand_dims(x, axis=0)).reshape(28, 28) # or just use [x] in model.predict(): 
																		 # we need to input a matrix, not a vector
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
