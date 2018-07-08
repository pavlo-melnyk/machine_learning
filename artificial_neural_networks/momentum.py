import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle
from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_W2, derivative_b2, derivative_W1, derivative_b1
from datetime import datetime


def main():
	# compare 3 scenarios:
	# 1. mini-batch SGD
	# 2. mini-batch SGD with the standard momentum
	# 3. mini-batch SGD with the Nesterov momentum

	max_iter = 20 
	print_period = 10

	X, Y = get_normalized_data(show_img=True)
	lr = 0.00004
	reg = 0.01

	Xtrain, Ytrain = X[:-1000, :], Y[:-1000]
	Xtest, Ytest = X[-1000:,:], Y[-1000:]
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)

	N, D = Xtrain.shape
	M = 300 # number of hidden layer units
	K = len(set(Ytrain))
	batch_size = 500
	n_batches = N // batch_size 

	# randomly initialize weights:
	W1 = np.random.randn(D, M) / np.sqrt(D)
	b1 = np.zeros(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.zeros(K)

	# save initial weights:
	W1_0 = W1.copy()
	b1_0 = b1.copy()
	W2_0 = W2.copy()
	b2_0 = b2.copy()
	
	
	# 1. batch GD:
	LL_batch = []
	CR_batch = []
	t0 = datetime.now()
	print('\nperforming batch SGD...')
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_size:(j+1)*batch_size, :]
			Ybatch = Ytrain_ind[j*batch_size:(j+1)*batch_size, :]
			p_Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)
			#print(Z.shape, p_Ybatch.shape, Ybatch.shape)
			#print('First batch cost:', cost(p_Ybatch, Ybatch))

			# updates:
			W2 -= lr*(derivative_W2(Z, Ybatch, p_Ybatch) + reg*W2)
			b2 -= lr*(derivative_b2(Ybatch, p_Ybatch) + reg*b2)
			W1 -= lr*(derivative_W1(Xbatch, Z, Ybatch, p_Ybatch, W2) + reg*W1)
			b1 -= lr*(derivative_b1(Z, Ybatch, p_Ybatch, W2) + reg*b1)

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				#print('pY:', pY)
				ll = cost(pY, Ytest_ind)
				LL_batch.append(ll)
				print('\ni: %d, j: %d, cost: %.6f' % (i, j, ll))

				error = error_rate(pY, Ytest)
				CR_batch.append(error)
				print('error rate:', error)

	dt1 = datetime.now() - t0
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('\nFinal error rate:', error_rate(pY, Ytest))
	print('Elapsed time:', dt1)

	# plot the cost
	'''
	plt.plot(LL_batch)
	plt.title('Cost for batch SGD')
	plt.show()
	'''


	# 2. batch SGD with momentum:
	# a) standart momentum ver #1
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()
	mu = 0.9 # momentum
	LL_momentum = []
	CR_momentum = []
	t0 = datetime.now()

	# initialize the weights' deltas as 0's,
	# 'cause we want to use them on the first iteration
	# before first updating
	dW2 = 0
	db2 = 0
	dW1 = 0
	db1 = 0
	print('performing batch SGD with the standart momentum ...')
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_size:(j+1)*batch_size, :]
			Ybatch = Ytrain_ind[j*batch_size:(j+1)*batch_size, :]           
			p_Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# updates:
			dW2 = mu*dW2 - lr*(derivative_W2(Z, Ybatch, p_Ybatch) + reg*W2)
			W2 += dW2
			db2 = mu*db2 - lr*(derivative_b2(Ybatch, p_Ybatch) + reg*b2)
			b2 += db2
			dW1 = mu*dW1 - lr*(derivative_W1(Xbatch, Z, Ybatch, p_Ybatch, W2) + reg*W1)
			W1 += dW1
			db1 = mu*db1 - lr*(derivative_b1(Z, Ybatch, p_Ybatch, W2) + reg*b1)
			b1 += db1

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_momentum.append(ll)
				print('\ni: %d, j: %d, cost: %.6f' % (i, j, ll))

				error = error_rate(pY, Ytest)
				CR_momentum.append(error)
				print('error rate:', error)

	dt2 = datetime.now() - t0
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('\nFinal error rate:', error_rate(pY, Ytest))
	print('Elapsed time:', dt2)

	# plot the cost
	'''
	plt.plot(LL_momentum)
	plt.title('Cost for batch SGD with the standart momentum')
	plt.show()
	'''

	'''
	# b) standart momentum ver #2
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()
	mu = 0.9 # momentum
	LL_momentum2 = []
	CR_momentum2 = []
	t0 = datetime.now()

	# initialize the weights' deltas as 0's,
	# 'cause we want to use them on the first iteration
	# before first updating
	dW2 = 0
	db2 = 0
	dW1 = 0
	db1 = 0
	print('performing batch SGD with the standart momentum v.2 ...')
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_size:(j+1)*batch_size, :]
			Ybatch = Ytrain_ind[j*batch_size:(j+1)*batch_size, :]           
			p_Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# updates:
			dW2 = mu*dW2 + (1 - mu)*(derivative_W2(Z, Ybatch, p_Ybatch) + reg*W2)
			W2 -= lr*dW2
			db2 = mu*db2 + (1 - mu)*(derivative_b2(Ybatch, p_Ybatch) + reg*b2)
			b2 -= lr*db2
			dW1 = mu*dW1 + (1 - mu)*(derivative_W1(Xbatch, Z, Ybatch, p_Ybatch, W2) + reg*W1)
			W1 -= lr*dW1
			db1 = mu*db1 + (1 - mu)*(derivative_b1(Z, Ybatch, p_Ybatch, W2) + reg*b1)
			b1 -= lr*db1

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_momentum2.append(ll)
				print('\ni: %d, j: %d, cost: %.6f' % (i, j, ll))

				error = error_rate(pY, Ytest)
				CR_momentum2.append(error)
				print('error rate:', error)

	dt4 = datetime.now() - t0
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('\nFinal error rate:', error_rate(pY, Ytest))
	print('Elapsed time:', dt4)

	# plot the cost
	
	plt.plot(LL_momentum2)
	plt.title('Cost for batch SGD with the standart momentum v.2')
	plt.show()
	
	'''

	# 3. batch GD with Nesterov momentum:
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()
	mu = 0.9 # momentum
	LL_nesterov_m = []
	CR_nesterov_m = []
	t0 = datetime.now()
	
	dW2 = 0
	db2 = 0
	dW1 = 0
	db1 = 0
	'''
	vW2 = 0
	vb2 = 0
	vW1 = 0
	vb1 = 0
	'''
	print('performing batch SGD with Nesterov momentum...')
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_size:(j+1)*batch_size, :]
			Ybatch = Ytrain_ind[j*batch_size:(j+1)*batch_size, :]
			p_Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)
			
			
			# updates:
			dW2 = mu*mu*dW2 - (mu + 1)*lr*(derivative_W2(Z, Ybatch, p_Ybatch) + reg*W2)
			W2 += dW2
			db2 = mu*mu*db2 - (mu + 1)*lr*(derivative_b2(Ybatch, p_Ybatch) + reg*b2)
			b2 += db2
			dW1 = mu*mu*dW1 - (mu + 1)*lr*(derivative_W1(Xbatch, Z, Ybatch, p_Ybatch, W2) + reg*W1)
			W1 += dW1
			db1 = mu*mu*db1 - (mu + 1)*lr*(derivative_b1(Z, Ybatch, p_Ybatch, W2) + reg*b1)
			b1 += db1
			
			'''
			# or the same, but step by step:
			# calculate gradients:
			gW2 = derivative_W2(Z, Ybatch, p_Ybatch) + reg*W2)
			gb2 = derivative_b2(Ybatch, p_Ybatch) + reg*b2)
			gW1 = derivative_W1(Xbatch, Z, Ybatch, p_Ybatch, W2) + reg*W1)
			gb1 = derivative_b1(Z, Ybatch, p_Ybatch, W2) + reg*b1)
			
			# v updates:
			vW2 = mu*vW2 - lr*gW2
			vb2 = mu*vb2 - lr*gb2
			vW1 = mu*vW1 - lr*gW1
			vb1 = mu*vb1 - lr*gb1

			# weights updates:
			W2 += mu*vW2 - lr*gW2
			b2 += mu*vb2 - lr*gb2
			W1 += mu*vW1 - lr*gW1
			b1 += mu*vb1 - lr*gb1
			'''

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_nesterov_m.append(ll)
				print('\ni: %d, j: %d, cost: %.6f' % (i, j, ll))

				error = error_rate(pY, Ytest)
				CR_nesterov_m.append(error)
				print('error rate:', error)

	dt3 = datetime.now() - t0
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('\nFinal error rate:', error_rate(pY, Ytest))
	print('Elapsed time:', dt3)

	# plot the cost
	'''
	plt.plot(LL_nesterov_m)
	plt.title('Cost for batch SGD with Nesterov momentum')
	plt.show()
	'''

	# plot the costs together:   
	plt.plot(LL_batch, label='batch')	
	plt.plot(LL_momentum, label='momentum')
	#plt.plot(LL_momentum2, label='momentum v2')	
	plt.plot(LL_nesterov_m, label='nesterov momentum')
	plt.xlabel('iterations')
	plt.ylabel('Cost')
	plt.legend()
	plt.show()
	


if __name__ == '__main__':
	main()





