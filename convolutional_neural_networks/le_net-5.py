import os
import numpy as np
import theano 
import theano.tensor as T 
import math
import matplotlib.pyplot as plt 

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from scipy.io import loadmat
from util import get_normalized_data
from sklearn.utils import shuffle

from datetime import datetime



def error_rate(y, t):
	return np.mean(y != t)


def swish(a):
	# betta = 1
	# return a / (1 + np.exp(-betta*a))
	# return a * (a > 0)
	return T.tanh(a)


def convpool(X, W, b, poolsize=(2, 2)):
	conv_out = conv2d(input=X, filters=W)

	# donwsample each feature map individually, using maxpooling:
	pooled_out = pool.pool_2d(
		input=conv_out,
		ws=poolsize,
		ignore_border=True
	)

	# add the bias term:
	# since the bias is a vector (1D array) - a value per feature map, 
	# we first reshape it to a tensor of shape (1, n_filters, 1, 1);
	# each bias will thus be broadcasted across mini-batches
	# and feature map width & height:
	return swish(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))


def init_filter(shape, poolsz):
	w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) / 2.0)
	return w.astype(np.float32)


def plot_filter(W, name='filter'):
	# create figure with number of subplots corresponding
	# to number of filters:
	N_filters = W.shape[0]
	N_channels = W.shape[1]
	n = math.ceil(math.sqrt(N_filters))

	fig, axes = plt.subplots(n, n)
	fig.suptitle(name, fontsize=16)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)
	
	# plot the weights:
	for i, ax in enumerate(axes.flat):
		for j in range(N_channels):
			if i < N_filters:
				img = W[i, j]
				ax.imshow(img, cmap='gray')
			
			ax.set_xticks([])
			ax.set_yticks([])
			
	plt.show()




def main():
	############################ step 1 ############################ 
	# load the data:
	X, Y = get_normalized_data(shape='theano', show_img=True)
	X, Y = X.astype(np.float32), Y.astype(np.int32)
	Xtrain, Ytrain = X[:-7000], Y[:-7000]
	Xvalid, Yvalid = X[-7000: -6000], Y[-7000:-6000]
	Xtest, Ytest = X[-6000:], Y[-6000:]

	# print the shapes of the sets:
	print('\nTraining set size:\nsamples: %d,  channels: %d,  sample size: %dx%d' % Xtrain.shape)
	print('Validation set size:\nsamples: %d,  channels: %d,  sample size: %dx%d' % Xvalid.shape)
	print('Test set size:\nsamples: %d,  channels: %d,  sample size: %dx%d' % Xtest.shape)

	N = Xtrain.shape[0]


	# define the hyperparameters:
	max_iter = 300
	print_period = 20

	lr = np.float32(5e-3)
	mu = np.float32(0.9)

	
	batch_size = 500
	n_batches = N // batch_size 

	# FC layer size:
	M1 = 120
	M2 = 84
	K = 10
	poolsz = (2, 2) # perhaps for later use

	# after the first conv out feature map will be of dimension 28 - 5 + 1 = 24
	# after downsample 24 / 2 = 12
	W1_shape = (6, 1, 5, 5) # (num_filters, num_color_channels, filter_width, filter_height)
	W1_init = init_filter(W1_shape, poolsz)
	b1_init = np.zeros(W1_shape[0], dtype=np.float32) # recall: one bias per feature map

	# after the second conv out feature map will be of dimension 12 - 5 + 1 = 8
	# after downsample 8 / 2 = 4
	W2_shape = (16, 6, 5, 5) # (num_filters, num_inp_feature_maps, filter_width, filter_height)
	W2_init = init_filter(W2_shape, poolsz)
	b2_init = np.zeros(W2_shape[0], dtype=np.float32) # recall: one bias per feature map

	# vanilla ANN weights (using "He Normal" initialization):
	W3_init = (np.random.randn(W2_shape[0]*4*4, M1) / np.sqrt(W2_shape[0]*4*4 / 2.0)).astype(np.float32)
	b3_init = np.zeros(M1, dtype=np.float32)
	W4_init = (np.random.randn(M1, M2) / np.sqrt(M1 / 2.0)).astype(np.float32)
	b4_init = np.zeros(M2, dtype=np.float32)
	W5_init = (np.random.randn(M2, K) / np.sqrt(M1 / 2.0)).astype(np.float32)
	b5_init = np.zeros(K, dtype=np.float32)
	# W4_init = (np.random.randn(M1, K) / np.sqrt(M1 / 2.0)).astype(np.float32)
	# b4_init = np.zeros(K, dtype=np.float32)


	############################ step 2 ############################
	# define theano variables and expressions:
	thX = T.tensor4('X', dtype='float32')
	thY = T.ivector('T')
	W1 = theano.shared(W1_init, 'W1')
	b1 = theano.shared(b1_init, 'b1')
	W2 = theano.shared(W2_init, 'W2')
	b2 = theano.shared(b2_init, 'b2')
	W3 = theano.shared(W3_init, 'W3')
	b3 = theano.shared(b3_init, 'b3')
	W4 = theano.shared(W4_init, 'W4')
	b4 = theano.shared(b4_init, 'b4')
	W5 = theano.shared(W5_init, 'W5')
	b5 = theano.shared(b5_init, 'b5')

	# forward pass:
	Z1 = convpool(thX, W1, b1)
	Z2 = convpool(Z1, W2, b2)
	Z3 = swish(Z2.flatten(ndim=2).dot(W3) + b3)
	Z4 = swish(Z3.dot(W4) + b4)
	pY = T.nnet.softmax(Z4.dot(W5) + b5)
	# pY = T.nnet.softmax(Z3.dot(W4) + b4)
	

	# define the cost function and prediction:
	cost = -(T.log(pY[T.arange(thY.shape[0]), thY])).mean()
	prediction = T.argmax(pY, axis=1)


	############################ step 3 ############################
	params = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5]
	# params = [W1, b1, W2, b2, W3, b3, W4, b4]

	# momentum deltas (of the same size as corresponding parameter =
	# weight matrix or bias vector - so we use .get_value method to
	# take the actual value of a parameter, because the parameter itself
	# is just a theano variable, and then use it to create a zeros_like 
	# numpy array of the same size):
	dparams = [theano.shared(np.zeros_like(p.get_value(), dtype=np.float32)) for p in params]

	# cost gradients wrt to parameters:
	grads = T.grad(cost, params)

	# collect the updates:
	updates = []
	for p, dp, g in zip(params, dparams, grads):
		dp_update = mu*dp - lr*g
		p_update = p + dp_update

		updates.append((dp, dp_update))
		updates.append((p, p_update))

	# define train and get_prediction function:
	train = theano.function(
		inputs=[thX, thY],
		updates=updates, 
	)

	get_prediction = theano.function(
		inputs=[thX, thY],
		outputs=[cost, prediction],
	)

	# the training loop:
	t0 = datetime.now()
	costs = []
	for i in range(max_iter):
		Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_size:(j+1)*batch_size, :]
			Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]

			train(Xbatch, Ybatch)
			if j % print_period == 0:
				cost_val, prediction_val = get_prediction(Xvalid, Yvalid)
				error = error_rate(prediction_val, Yvalid)
				print('\ni: %d,  j: %d,  valid_cost: %.3f,  error: %.3f' % (i, j, cost_val, error))
				costs.append(cost_val)

	print('\nElapsed time: ', datetime.now() - t0)
	_, train_prediction_val = get_prediction(Xtrain, Ytrain)
	_, test_prediction_val = get_prediction(Xtest, Ytest)
	print('\nTraininig set accuracy: %.6f' % (1 - error_rate(train_prediction_val, Ytrain)))
	print('Test set accuracy: %.6f' % (1 - error_rate(test_prediction_val, Ytest)))

	plt.plot(costs)
	plt.title('Cost on Validation Set')
	plt.xlabel('iterations')
	plt.ylabel('cost')
	plt.show()


	############################ step 4 ############################
	# visualize learned filters:
	plot_filter(W1.get_value(), 'W1')
	plot_filter(W2.get_value(), 'W2')




if __name__ == "__main__":
	main()