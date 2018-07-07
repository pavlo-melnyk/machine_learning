import numpy as np 
import theano
import theano.tensor as T
import matplotlib.pyplot as plt 
import joblib

from theano.tensor.nnet.bn import batch_normalization_train, batch_normalization_test
from util import get_normalized_data, error_rate, relu, swish, init_weight_and_bias
from sklearn.utils import shuffle
from datetime import datetime 



def init_weight(M1, M2):
	return np.random.randn(M1, M2) / np.sqrt(M1 / 2.0)

	
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



class HiddenLayerBatchNorm(object):
	def __init__(self, M1, M2, an_id, f):
		self.id = an_id
		self.f = f # activation function
		self.M1 = M1
		self.M2 = M2

		W = init_weight(M1, M2).astype(np.float32)
		gamma = np.ones(M2).astype(np.float32)
		betta = np.zeros(M2).astype(np.float32)


		self.W = theano.shared(W, name='W%s' % self.id)
		self.gamma = theano.shared(gamma, name='gamma%s' % self.id)
		self.betta = theano.shared(betta, name='betta%s' % self.id)
		
		# for test step we are to need a running-mean and a running-variance:
		self.rn_mean = theano.shared(np.zeros(M2).astype(np.float32))
		self.rn_var = theano.shared(np.zeros(M2).astype(np.float32))

		self.params = [self.W, self.gamma, self.betta]

	def forward(self, X, is_training, decay=0.9):
		Z = X.dot(self.W)

		if is_training:
			Z, batch_mean, batch_invstd, new_rn_mean, new_rn_var = batch_normalization_train(
				Z, self.gamma, self.betta, running_mean=self.rn_mean, running_var=self.rn_var)

			self.rn_update = [(self.rn_mean, new_rn_mean), (self.rn_var, new_rn_var)]
		
		else: 
			Z = batch_normalization_test(
				Z, self.gamma, self.betta, self.rn_mean, self.rn_var)

		return self.f(Z)
	


class HiddenLayer(object):
	def __init__(self, M1, M2, an_id, f):
		self.id = an_id
		self.f = f # activation function
		self.M1 = M1
		self.M2 = M2
		W, b = init_weight_and_bias(M1, M2)
		self.W = theano.shared(W, 'W%s' % self.id)
		self.b = theano.shared(b, 'b%s' % self.id)
		self.params = [self.W, self.b]


	def forward(self, X):
		return self.f(X.dot(self.W) + self.b)



class ANN(object):
	def __init__(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes


	def fit(self, X, Y, learning_rate=1e-5, mu=0.99, decay=0.999, epochs=100, batch_sz=100, reg=0.0, eps=10e-10, display_cost=False):
		t0 = datetime.now()

		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)
		reg = np.float32(reg)
		eps = np.float32(eps)

		# first, make a validation set:	
		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)	
		Y = Y.astype(np.int32)
		Xvalid, Yvalid = X[-1000:, :], Y[-1000:]
		X, Y = X[:-1000, :], Y[:-1000]

		# initialize the hidden layers:
		N, D = X.shape
		K = len(set(Y))
		self.layers = []

		# first dimension size of a first matrix:
		M1 = D
		count = 0 # for the id of the weigts/biases
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayerBatchNorm(M1, M2, count, swish)
			self.layers.append(h)
			M1 = M2 # update the first dimension size fir the next iteration
			count += 1

		# for the last weight/bias matrix (vector):
		h = HiddenLayer(M1, K, count, T.nnet.softmax)
		self.layers.append(h)

		# collect all the parameters we are going to use during Gradient Descent:
		self.parameters = []
		for h in self.layers[::-1]:
			self.parameters += h.params 
		
		# in order to use Momentum,
		# we are to keep track of all the changes (dW's and db's):
		dparams = [theano.shared(np.zeros_like(p.get_value(), dtype=np.float32)) for p in self.parameters]

		# define theano variables and functions:
		thX = T.matrix('X')
		thY = T.ivector('Y') # a vector of integers

		# for training:
		pY = self.forward(thX, is_training=True)
		#regularization = l2*T.sum([(p*p).sum() for p in self.parameters])
		cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY]))
		prediction = T.argmax(pY, axis=1) # will do sort of T.argmax(pY, axis=1)
		
		# for test step:
		test_pY = self.forward(thX, is_training=False)
		test_cost = -T.mean(T.log(test_pY[T.arange(thY.shape[0]), thY]))
		test_prediction = T.argmax(test_pY, axis=1)
		cost_predict_op = theano.function(
			inputs = [thX, thY], 
			outputs = [test_cost, test_prediction]
		)

		# the updates for the train function:
		# just for momentum:
		
		# updates = [
		# 	(dp, mu*dp - learning_rate*T.grad(cost, p)) for dp, p in zip(dparams, self.parameters)
		# ] + [
		# 	(p, p + dp) for p, dp in zip(self.parameters, dparams)
		# ]
		

		# for RMSProp with momentum:		
		updates = rmsprop(cost, self.parameters, learning_rate, mu, decay, eps)
		
		# don't forget to collect rn_mean and rn_var parameters of every hidden layer
		# for updating:
		for layer in self.layers[:-1]:
			updates += layer.rn_update		


		train_op = theano.function(
			inputs=[thX, thY],
			updates=updates
		)

		self.predict = theano.function(
			inputs=[thX],
			outputs=[test_prediction]
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

		dt = datetime.now() - t0

		if display_cost:
			plt.plot(costs)
			plt.show()

		print('\nElapsed time:', dt)
		
		train_acc = self.score(X, Y)
		valid_acc = self.score(Xvalid, Yvalid)		
		print('\nTrain set acc:', train_acc, ' Valid set acc:', valid_acc)


	def forward(self, X, is_training):
		Z = X
		for h in self.layers[:-1]:
			Z = h.forward(Z, is_training)
		out = self.layers[-1].forward(Z)
		return out


	def score(self, X, Y):
		prediction = self.predict(X)
		return np.mean(Y == prediction)




def main():	
	X, Y = get_normalized_data() # normalized MNIST dataset
	Xtest, Ytest = X[-1000:, :], Y[-1000:]
	Xtrain, Ytrain = X[:-1000, :], Y[:-1000]

	model = ANN([1000, 500, 500])
	model.fit(Xtrain, Ytrain, display_cost=True)
	# joblib.dump(model, 'mymodel.pkl')
	# model = joblib.load('mymodel.pkl')
	print('Test set acc:', model.score(Xtest.astype(np.float32), Ytest.astype(np.float32)))





if __name__ == '__main__':
	main()