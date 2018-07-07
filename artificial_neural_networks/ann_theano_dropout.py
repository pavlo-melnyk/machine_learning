import numpy as np 
import theano
import theano.tensor as T
import matplotlib.pyplot as plt 

from util import get_normalized_data, error_rate, relu, swish, init_weight_and_bias
from sklearn.utils import shuffle
from theano.tensor.shared_randomstreams import RandomStreams
from datetime import datetime 



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



class HiddenLayer(object):
	def __init__(self, M1, M2, an_id):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2
		W, b = init_weight_and_bias(M1, M2)
		self.W = theano.shared(W, 'W%s' % self.id)
		self.b = theano.shared(b, 'b%s' % self.id)
		self.params = [self.W, self.b]


	def forward(self, X):
		return swish(X.dot(self.W) + self.b)



class ANN(object):
	def __init__(self, hidden_layer_sizes, p_keep):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.dropout_rates = p_keep


	def fit(self, X, Y, learning_rate=10e-5, mu=0.9, decay=0.99, epochs=10, batch_sz=100, eps=10e-10, display_cost=False):
		#learning_rate=10e-7, mu=0.99, decay=0.999, epochs=100, batch_sz=30, l2=0.0, eps=10e-10
		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)		
		eps = np.float32(eps)
		'''
		In Theano we can't actually 'drop' the nodes;
		that would result in a different computational graph,
		we are instead to multiply nodes by 1 and 0;
		for each layer we then need to create a 'mask' - array of 0s and 1s;
		Theano graph nodes don't have values, so we can't multiply them by numpy array 'mask';
		instead we want Theano to generate random values every time it's called;
		thus we create an instance of RandomStreams object:
		'''
		self.rng = RandomStreams()

		# first, make a validation set:	
		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)	
		Y = Y.astype(np.int32)
		Xvalid, Yvalid = X[-1000:, :], Y[-1000:]
		X, Y = X[:-1000, :], Y[:-1000]

		

		#initialize the hidden layers:
		N, D = X.shape
		K = len(set(Y))
		self.hidden_layers = []

		# the size of the first dimension of the first matrix:
		M1 = D
		count = 0 # for the id of the weigts/biases
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2, count)
			self.hidden_layers.append(h)
			M1 = M2 # update the first dimension size fir the next iteration
			count += 1

		# for the last weight/bias matrix (vector):
		W, b = init_weight_and_bias(M1, K)
		self.W = theano.shared(W, 'W%s' % count)
		self.b = theano.shared(b, 'b%s' % count)

		# collect all the parameters we are going to use during Gradient Descent:
		self.parameters = [self.W, self.b]
		for h in self.hidden_layers[::-1]:
			self.parameters += h.params 
		
		# in order to use Momentum,
		# we are to keep track of all the changes (dW's and db's):
		dparams = [theano.shared(np.zeros_like(p.get_value(), dtype=np.float32)) for p in self.parameters]

		# for RMSProp,
		# we are to keep track of caches (cache_W's and cache_b's) as well:
		caches = [theano.shared(np.ones_like(p.get_value(), dtype=np.float32)) for p in self.parameters]

		# define theano variables and functions:
		thX = T.matrix('X')
		thY = T.ivector('Y') # a vector of integers

		# since we do dropout, we drop the nodes only on training step,
		# when evaluating we just scale them;
		# so we need to define two expressions for the output and cost calculations:
		pY_train = self.forward_train(thX)
		pY_predict = self.forward_predict(thX)

		
		cost = -T.mean(T.log(pY_train[T.arange(thY.shape[0]), thY]))
		cost_predict = -T.mean(T.log(pY_predict[T.arange(thY.shape[0]), thY]))
		
		prediction = self.predict(thX) # will do sort of T.argmax(pY, axis=1)

		cost_predict_op = theano.function(
			inputs = [thX, thY], 
			outputs = [cost_predict, prediction]
		)

		# the updates for the train function:
		
		updates = [
			(cache, decay*cache + (np.float32(1.0)-decay)*T.grad(cost, p)**2) for p, cache in zip(self.parameters, caches)
		] + [
			(dp, mu*dp - learning_rate*T.grad(cost, p)/T.sqrt(cache + eps)) for dp, p, cache in zip(dparams, self.parameters, caches)
		] + [
			(p, p + dp) for p, dp in zip(self.parameters, dparams)
		]
		
		#updates = rmsprop(cost, self.parameters, learning_rate, mu, decay, eps)

		train_op = theano.function(
			inputs=[thX, thY],
			updates=updates
		)

		# batch SGD:
		n_batches = N // batch_sz
		costs = []
		for i in range(epochs):
			X, Y = shuffle(X, Y)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz : (j+1)*batch_sz, :]
				Ybatch = Y[j*batch_sz : (j+1)*batch_sz]

				train_op(Xbatch, Ybatch)

				if j % 20 == 0:
					c, p = cost_predict_op(Xvalid, Yvalid)
					costs.append(c)
					e = error_rate(Yvalid, p)
					print('\ni: %d,  j: %d, cost: %.6f, \nerror: %.6f' % (i, j, c, e))

		if display_cost:
			plt.plot(costs)
			plt.show()


	def forward_train(self, X):
		'''
		Used for training; does forward propagation with dropout.
		'''
		Z = X		
		for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):
			mask = self.rng.binomial(n=1, p=p, size=Z.shape) # create a 'mask'
			Z *= mask # elementwise multiplication - 'droping' the nodes
			Z = h.forward(Z)			
		# for the last hidden layer:
		mask = self.rng.binomial(n=1, p=self.dropout_rates[-1], size=Z.shape)
		Z *= mask
		return T.nnet.softmax(Z.dot(self.W) + self.b)


	def predict(self, X):
		pY = self.forward_predict(X)
		return T.argmax(pY, axis=1)


	def forward_predict(self, X):
		'''
		Used in prediction operation; 
		does scaling nodes operation by multiplying each node 
		by corresponding p_keep.
		'''
		Z = X		
		for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):
			Z = h.forward(Z * p)			
		# for the last hidden layer:
		Z *= self.dropout_rates[-1]
		return T.nnet.softmax(Z.dot(self.W) + self.b)



def main():	
	X, Y = get_normalized_data() # normalize MNIST dataset
	t0 = datetime.now()
	model = ANN([2000, 1000], [0.8, 0.5, 0.5])
	model.fit(X, Y, display_cost=True)
	dt = datetime.now() - t0

	print('Elapsed time:', dt)



if __name__ == '__main__':
	main()