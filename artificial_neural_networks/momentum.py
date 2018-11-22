import os
import sys
import numpy as np
import matplotlib.pyplot as plt 

from mlp import forward, derivative_W2, derivative_b2, derivative_W1, derivative_b1
from util import error_rate, cross_entropy, cost

from sklearn.utils import shuffle
from datetime import datetime
from keras.datasets import mnist



def y2indicator(y):
	N, K = len(y), len(set(y))
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, y[i]] = 1
	return ind


def main():
	# load the data:
	(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
	# print(Xtrain.shape)
	N, d, _ = Xtrain.shape
	D = d*d
	Ntest = len(Xtest)

	# normalize the data:
	Xtrain = Xtrain / 255.0
	Xtest = Xtest / 255.0

	# display:
	# n = np.random.choice(N)
	# plt.imshow(Xtrain[n], cmap='gray')
	# plt.title(str(Ytrain[n]))
	# plt.show()

	# reshape the data:
	Xtrain = Xtrain.reshape(N, D)
	Xtest = Xtest.reshape(Ntest, D)	

	# print('Xtrain.max():', Xtrain.max())
	# print('Xtrain.shape:', Xtrain.shape)

	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)


	# define hyperparameters:
	epochs = 30
	print_period = 10
	lr = 0.00004
	reg = 0.01

	batch_sz = 500
	n_batches = N // batch_sz

	M = 300 # the hidden layer size
	K = len(set(Ytrain))

	# randomly initialize the weights:
	W1_init = np.random.randn(D, M) / np.sqrt(D)
	b1_init = np.zeros(M)
	W2_init = np.random.randn(M, K) / np.sqrt(M)
	b2_init = np.zeros(K)

	
	# 1. mini-batch SGD:
	losses_batch = []
	errors_batch = []

	W1 = W1_init.copy()
	b1 = b1_init.copy()
	W2 = W2_init.copy()
	b2 = b2_init.copy()

	print('\nmini-batch SGD')

	t0 = datetime.now()
	for i in range(epochs):
		Xtrain, Ytrain_ind = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
			Ybatch = Ytrain_ind[j*batch_sz:(j+1)*batch_sz]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# update the params:
			W2 -= lr*(derivative_W2(Z, Ybatch, pYbatch) + reg*W2)
			b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
			W1 -= lr*(derivative_W1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
			b1 -= lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				l = cross_entropy(pY, Ytest)
				losses_batch.append(l)
				e = error_rate(pY, Ytest)
				errors_batch.append(e)
				sys.stdout.write('epoch: %d, batch: %d, cost: %.6f, error_rate: %.4f\r' % (i, j, l, e))
				# print('\nepoch: %d, batch: %d, cost: %6f' % (i, j, l))
				# print('error_rate:', e)

	sys.stdout.flush()	
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('ETA:', datetime.now() - t0, 'final error rate:', error_rate(pY, Ytest), ' '*20)

	
	# 2. mini-batch SGD with momentum - version 1:
	losses_momentum1 = []
	errors_momentum1 = []

	W1 = W1_init.copy()
	b1 = b1_init.copy()
	W2 = W2_init.copy()
	b2 = b2_init.copy()

	mu = 0.9 # momentum term
	# initial values for the 'velocities':
	dW2 = 0 
	db2 = 0
	dW1 = 0
	db1 = 0

	print('\nmini-batch SGD with momentum - version 1')
	t0 = datetime.now()
	for i in range(epochs):
		Xtrain, Ytrain_ind = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
			Ybatch = Ytrain_ind[j*batch_sz:(j+1)*batch_sz]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# calculate the gradients:
			gW2 = derivative_W2(Z, Ybatch, pYbatch) + reg*W2
			gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
			gW1 = derivative_W1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
			gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1

			# update the 'velocities':
			dW2 = mu*dW2 - lr*gW2  
			db2 = mu*db2 - lr*gb2 
			dW1 = mu*dW1 - lr*gW1 
			db1 = mu*db1 - lr*gb1 
			
			# update the params:
			W2 += dW2
			b2 += db2
			W1 += dW1
			b1 += db1

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				l = cross_entropy(pY, Ytest)
				losses_momentum1.append(l)
				e = error_rate(pY, Ytest)
				errors_momentum1.append(e)
				sys.stdout.write('epoch: %d, batch: %d, cost: %.6f, error_rate: %.4f\r' % (i, j, l, e))
				# print('\nepoch: %d, batch: %d, cost: %6f' % (i, j, l))
				# print('error_rate:', e)
	
	sys.stdout.flush()
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('ETA:', datetime.now() - t0, 'final error rate:', error_rate(pY, Ytest), ' '*20)
	
	'''
	# 3. mini-batch SGD with momentum - version 2:
	losses_momentum2 = []
	errors_momentum2 = []

	W1 = W1_init.copy()
	b1 = b1_init.copy()
	W2 = W2_init.copy()
	b2 = b2_init.copy()

	mu = 0.9 # momentum term
	# initial values for the 'velocities':
	dW2 = 0 
	db2 = 0
	dW1 = 0
	db1 = 0

	# lr = 0.0004

	print('\nmini-batch SGD with momentum - version 2')
	t0 = datetime.now()
	for i in range(epochs):
		Xtrain, Ytrain_ind = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
			Ybatch = Ytrain_ind[j*batch_sz:(j+1)*batch_sz]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# calculate the gradients:
			gW2 = derivative_W2(Z, Ybatch, pYbatch) + reg*W2
			gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
			gW1 = derivative_W1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
			gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1

			# # update the 'velocities':
			dW2 = mu*dW2 + (1-mu)*gW2  
			db2 = mu*db2 + (1-mu)*gb2 
			dW1 = mu*dW1 + (1-mu)*gW1 
			db1 = mu*db1 + (1-mu)*gb1 

			# update the 'velocities':
			# dW2 = mu*dW2 + gW2  
			# db2 = mu*db2 + gb2 
			# dW1 = mu*dW1 + gW1 
			# db1 = mu*db1 + gb1 
			
			# update the params:
			W2 -= lr*dW2
			b2 -= lr*db2
			W1 -= lr*dW1
			b1 -= lr*db1

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				l = cross_entropy(pY, Ytest)
				losses_momentum2.append(l)
				e = error_rate(pY, Ytest)
				errors_momentum2.append(e)
				sys.stdout.write('epoch: %d, batch: %d, cost: %.6f, error_rate: %.4f\r' % (i, j, l, e))
				sys.stdout.flush()				
				
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('ETA:', datetime.now() - t0, 'final error rate:', error_rate(pY, Ytest), ' '*20)
    # best result: epochs = 25, final_error = 0.0179
	'''
	# 4. mini-batch SGD with Nesterov momentum:
	losses_nesterov_momentum = []
	errors_nesterov_momentum = []

	W1 = W1_init.copy()
	b1 = b1_init.copy()
	W2 = W2_init.copy()
	b2 = b2_init.copy()

	mu = 0.9 # momentum term
	# initial values for the 'velocities':
	dW2 = 0 
	db2 = 0
	dW1 = 0
	db1 = 0


	print('\nmini-batch SGD with Nesterov momentum')
	t0 = datetime.now()
	for i in range(epochs):
		Xtrain, Ytrain_ind = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
			Ybatch = Ytrain_ind[j*batch_sz:(j+1)*batch_sz]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# calculate the gradients:
			gW2 = derivative_W2(Z, Ybatch, pYbatch) + reg*W2
			gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
			gW1 = derivative_W1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
			gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1
			
			# update the 'velocities':
			dW2 = mu*dW2 - lr*gW2  
			db2 = mu*db2 - lr*gb2 
			dW1 = mu*dW1 - lr*gW1 
			db1 = mu*db1 - lr*gb1 
			
			# update the params:
			W2 += mu*dW2 - lr*gW2  
			b2 += mu*db2 - lr*gb2 
			W1 += mu*dW1 - lr*gW1 
			b1 += mu*db1 - lr*gb1 

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				l = cross_entropy(pY, Ytest)
				losses_nesterov_momentum.append(l)
				e = error_rate(pY, Ytest)
				errors_nesterov_momentum.append(e)
				sys.stdout.write('epoch: %d, batch: %d, cost: %.6f, error_rate: %.4f\r' % (i, j, l, e))
				sys.stdout.flush()
				# print('\nepoch: %d, batch: %d, cost: %6f' % (i, j, l))
				# print('error_rate:', e)

	
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('ETA:', datetime.now() - t0, 'final error rate:', error_rate(pY, Ytest), ' '*20)
	
	# plot the losses:
	plt.plot(losses_batch, label='mini-batch SGD')
	plt.plot(losses_momentum1, label='+ momentum')
	plt.plot(losses_nesterov_momentum, label='+ Nesterov momentum')
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()

