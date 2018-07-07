import numpy as np 
import theano
import theano.tensor as T
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
from datetime import datetime

from util import getImageData, error_rate, init_weight_and_bias, init_filter_, swish



def rmsprop(cost, params, lr, mu, decay, eps):
	grads = T.grad(cost, params)
	updates = []
	for p, g in zip(params, grads):
		# cache
		ones = np.ones_like(p.get_value(), dtype=np.float32)
		# initialize cache to 1:
		c = theano.shared(ones)
		new_c = decay*c + (np.float32(1.0) - decay)*g*g

		# momentum:
		zeros = np.zeros_like(p.get_value(), dtype=np.float32)
		# initialize velocity to 0:
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
		W0, b0 = init_weight_and_bias(M1, M2)
		self.W = theano.shared(W0, 'W%s' % self.id)
		self.b = theano.shared(b0, 'b%s' % self.id)
		self.params = [self.W, self.b]

	def forward(self, X):
		return swish(X.dot(self.W) + self.b)



class ConvPoolLayer(object):
	def __init__(self, mi, mo, fh=5, fw=5, border_mode='valid', s=(1,1), poolsz=(2, 2)):
		# mi = input feature map size;
		# mo = output feature map size;
		sz = (mo, mi, fh, fw)
		W0 = init_filter_(sz)
		self.W = theano.shared(W0)
		b0 = np.zeros(mo, dtype=np.float32)
		self.b = theano.shared(b0)
		self.border_mode = border_mode
		self.s = s      # stride
		self.poolsz = poolsz
		self.params = [self.W, self.b]

	def forward(self, X):
		conv_out = conv2d(
			input=X, 
			filters=self.W, 
			border_mode=self.border_mode,
			subsample=self.s,			
		)
		pooled_out = pool_2d(
			input=conv_out,
			ws=self.poolsz, 
			ignore_border=True, 
			mode='max',
		)
		return swish(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))



class CNN(object):
	def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
		self.convpool_layer_sizes = convpool_layer_sizes
		self.hidden_layer_sizes = hidden_layer_sizes

	def fit(self, X, Y, lr=1e-3, mu=0.99, reg=1e-3, decay=0.99999, eps=1e-10, batch_size=30, epochs=10, display_cost=False):
		lr = np.float32(lr)
		mu = np.float32(mu)
		reg = np.float32(reg)
		decay = np.float32(decay)
		eps = np.float32(eps)

		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)
		Y = Y.astype(np.int32)

		# create a validation set:
		Xvalid, Yvalid = X[-1000:,], Y[-1000:]
		X, Y = X[:-1000,], Y[:-1000]

		# initialize convpool layers:
		N, c, height, width = X.shape
		mi = c 
		outh = height
		outw = width
		self.convpool_layers = []
		for mo, fh, fw in self.convpool_layer_sizes:
			layer = ConvPoolLayer(mi, mo, fh, fw)
			self.convpool_layers.append(layer)
			# output volume height and width 
			# after the current convpool layer:
			outh = (outh - fh + 1) // 2
			outw = (outw - fh + 1) // 2
			mi = mo

		# initialize mlp layers:
		K = len(set(Y))
		self.hidden_layers = []
		# size must be the same as output of last convpool layer:
		M1 = self.convpool_layer_sizes[-1][0]*outh*outw
		count = 0 # will be used to id hidden layers
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2, count)
			self.hidden_layers.append(h)
			M1 = M2
			count += 1

		# the last layer - softmax output:
		W, b = init_weight_and_bias(M1, K)
		self.W = theano.shared(W, 'W_output')
		self.b = theano.shared(b, 'b_output')

		# collect params:
		self.params = []
		for layer in self.convpool_layers:
			self.params += layer.params
		for h_layer in self.hidden_layers:
			self.params += h_layer.params
		self.params += [self.W, self.b]

		# set up theano functions and variables:
		thX = T.tensor4('X', dtype='float32')
		thY = T.ivector('Y')
		pY = self.forward(thX) # the forward func will be defined

		cost = -T.mean(T.log(pY[T.arange(pY.shape[0]), thY]))
		# add the regularization term to the cost:
		reg_term = reg*T.sum([(p*p).sum() for p in self.params])
		cost += reg_term

		prediction = self.th_predict(thX)

		# theano function to make the actual calculation of cost
		# and get the prediction:
		cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

		updates = rmsprop(cost, self.params, lr, mu, decay, eps)
		train_op = theano.function(
			inputs=[thX, thY], 
			updates=updates,
			outputs=cost, 
		)

		# the training loop:
		n_batches = N // batch_size
		train_costs = []
		valid_costs = []
		t0 = datetime.now()
		for i in range(epochs):
			X, Y = shuffle(X, Y)
			for j in range(n_batches):
				Xbatch = X[j*batch_size:(j+1)*batch_size, :]
				Ybatch = Y[j*batch_size:(j+1)*batch_size]

				train_cost = train_op(Xbatch, Ybatch)
				train_costs.append(train_cost)
				if j % 20 == 0:
					cost_val, prediction_val = cost_predict_op(Xvalid, Yvalid)
					error = error_rate(prediction_val, Yvalid)
					print('\ni: %d,  j: %d,  valid_cost: %.3f,  error: %.3f' % (i, j, cost_val, error))
					valid_costs.append(cost_val)

		print('\nElapsed time: ', datetime.now() - t0)
			
		if display_cost:
			plt.plot(train_costs)
			plt.title('Cost on Training Set')
			plt.xlabel('iterations')
			plt.show()

			plt.plot(valid_costs)
			plt.title('Cost on Validation Set')
			plt.xlabel('iterations')
			plt.show()

	def forward(self, X):
		Z = X
		for layer in self.convpool_layers:
			Z = layer.forward(Z)
		Z = Z.flatten(ndim=2)
		for h_layer in self.hidden_layers:
			Z = h_layer.forward(Z)
		# self.W and self.b are the parameters of the last layer - softmax layer:
		return T.nnet.softmax(Z.dot(self.W) + self.b)

	def th_predict(self, X):
		pY = self.forward(X)
		return T.argmax(pY, axis=1)



def main():
	X, Y = getImageData()

	model = CNN(
		convpool_layer_sizes=[(32, 3, 3), (64, 3, 3), (128, 3, 3), (256, 3, 3)],
		hidden_layer_sizes=[500, 300],
		)
	model.fit(X, Y, display_cost=True)


if __name__ == '__main__':
	main()

