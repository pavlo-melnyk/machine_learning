import numpy as np 
import matplotlib.pyplot as plt 

from mlp import forward, derivative_W2, derivative_b2, derivative_W1, derivative_b1
from util import get_normalized_data, cost, y2indicator, error_rate
from sklearn.utils import shuffle
from datetime import datetime 


def main():
	# compare 5 scenarios:
	# 1. batch SGD with constant learning rate
	# 2. batch SGD with RMSProp
	# 3. batch SGD with AdaGrad
	# 4. batch SGD with exponential decay
	

	np.random.seed(2)

	max_iter = 20 
	print_period = 10

	X, Y = get_normalized_data()
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

	
	# 1. batch SGD with constant learning rate:
	LL_batch = []
	CR_batch = []
	t0 = datetime.now()
	print('\nperforming batch SGD with constant learning rate...')
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
	final_err1 = error_rate(pY, Ytest)


	# plot the cost
	#plt.plot(LL_batch)
	#plt.title('Cost for batch GD with const lr')
	#plt.show()



	# 2. batch GD with RMSProp:
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()

	LL_RMSProp = []
	CR_RMSProp = []

	lr0 = 0.001 # initial learning rate
	cache_W2 = 1
	cache_b2 = 1
	cache_W1 = 1
	cache_b1 = 1
	decay = 0.999
	eps = 10e-10

	t0 = datetime.now()

	print('\nperforming batch SGD with RMSProp...')
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_size:(j+1)*batch_size, :]
			Ybatch = Ytrain_ind[j*batch_size:(j+1)*batch_size, :]
			p_Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)
			#print(Z.shape, p_Ybatch.shape, Ybatch.shape)
			#print('First batch cost:', cost(p_Ybatch, Ybatch))

			# updates:
			gW2 = (derivative_W2(Z, Ybatch, p_Ybatch) + reg*W2)
			cache_W2 = decay*cache_W2 + (1 - decay)*gW2*gW2
			W2 -= lr0*gW2 / np.sqrt(cache_W2 + eps)
			
			gb2 = (derivative_b2(Ybatch, p_Ybatch) + reg*b2)
			cache_b2 = decay*cache_b2 + (1 - decay)*gb2*gb2
			b2 -= lr0*gb2 / np.sqrt(cache_b2 + eps)
			
			gW1 = (derivative_W1(Xbatch, Z, Ybatch, p_Ybatch, W2) + reg*W1)
			cache_W1 = decay*cache_W1 + (1 - decay)*gW1*gW1
			W1 -= lr0*gW1 / np.sqrt(cache_W1 + eps)
			
			gb1 = (derivative_b1(Z, Ybatch, p_Ybatch, W2) + reg*b1)
			cache_b1 = decay*cache_b1 + (1 - decay)*gb1*gb1
			b1 -= lr0*gb1 / np.sqrt(cache_b1 + eps)


			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				#print('pY:', pY)
				ll = cost(pY, Ytest_ind)
				LL_RMSProp.append(ll)
				print('\ni: %d, j: %d, cost: %.6f' % (i, j, ll))

				error = error_rate(pY, Ytest)
				CR_RMSProp.append(error)
				print('error rate:', error)

	dt2 = datetime.now() - t0
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	final_err2 = error_rate(pY, Ytest)
	

	# plot the cost
	#plt.plot(LL_RMSProp)
	#plt.title('Cost for batch SGD with RMSProp')
	#plt.show()


	# 3. batch SGD with AdaGrad:
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()

	LL_AdaGrad = []
	CR_AdaGrad = []

	lr0 = 0.01 # initial learning rate
	cache_W2 = 1
	cache_b2 = 1
	cache_W1 = 1
	cache_b1 = 1
	eps = 10e-10

	t0 = datetime.now()

	print('\nperforming batch SGD with AdaGrad...')
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_size:(j+1)*batch_size, :]
			Ybatch = Ytrain_ind[j*batch_size:(j+1)*batch_size, :]
			p_Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)
			#print(Z.shape, p_Ybatch.shape, Ybatch.shape)
			#print('First batch cost:', cost(p_Ybatch, Ybatch))

			# updates:
			gW2 = (derivative_W2(Z, Ybatch, p_Ybatch) + reg*W2)
			cache_W2 = cache_W2 + gW2*gW2
			W2 -= lr0*gW2 / np.sqrt(cache_W2 + eps)
			
			gb2 = (derivative_b2(Ybatch, p_Ybatch) + reg*b2)
			cache_b2 = cache_b2 + gb2*gb2
			b2 -= lr0*gb2 / np.sqrt(cache_b2 + eps)
			
			gW1 = (derivative_W1(Xbatch, Z, Ybatch, p_Ybatch, W2) + reg*W1)
			cache_W1 = cache_W1 + gW1*gW1
			W1 -= lr0*gW1 / np.sqrt(cache_W1 + eps)
			
			gb1 = (derivative_b1(Z, Ybatch, p_Ybatch, W2) + reg*b1)
			cache_b1 = cache_b1 + gb1*gb1
			b1 -= lr0*gb1 / np.sqrt(cache_b1 + eps)


			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				#print('pY:', pY)
				ll = cost(pY, Ytest_ind)
				LL_AdaGrad.append(ll)
				print('\ni: %d, j: %d, cost: %.6f' % (i, j, ll))

				error = error_rate(pY, Ytest)
				CR_AdaGrad.append(error)
				print('error rate:', error)

	dt3 = datetime.now() - t0
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	final_err3 = error_rate(pY, Ytest)


	# plot the cost
	#plt.plot(LL_AdaGrad)
	#plt.title('Cost for batch SGD with AdaGrad')
	#plt.show()
	'''

	# 4. batch SGD with exponential decay:
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()

	LL_exp = []
	CR_exp = []

	
	lr0 = 0.0004 # initial learning rate
	k = 1e-7
	t = 0 # initial log
	lr = lr0 
	t0 = datetime.now()

	print('\nperforming batch SGD with lr exponential decay...')
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_size:(j+1)*batch_size, :]
			Ybatch = Ytrain_ind[j*batch_size:(j+1)*batch_size, :]
			p_Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)
			#print(Z.shape, p_Ybatch.shape, Ybatch.shape)
			#print('First batch cost:', cost(p_Ybatch, Ybatch))
			
			# updates:
			gW2 = (derivative_W2(Z, Ybatch, p_Ybatch) + reg*W2)
			W2 -= lr*gW2
			
			gb2 = (derivative_b2(Ybatch, p_Ybatch) + reg*b2)			
			b2 -= lr*gb2
			
			gW1 = (derivative_W1(Xbatch, Z, Ybatch, p_Ybatch, W2) + reg*W1)
			W1 -= lr*gW1
			
			gb1 = (derivative_b1(Z, Ybatch, p_Ybatch, W2) + reg*b1)
			b1 -= lr*gb1 

			# decrease the learning rate
			lr = lr0 * np.exp(-k*t)
			t += 1

			if j % print_period == 0:
				print('current learning rate:', lr)
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				#print('pY:', pY)
				ll = cost(pY, Ytest_ind)
				LL_exp.append(ll)
				print('\ni: %d, j: %d, cost: %.6f' % (i, j, ll))

				error = error_rate(pY, Ytest)
				CR_exp.append(error)
				print('error rate:', error)

	dt4 = datetime.now() - t0
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('\nFinal error rate:', error_rate(pY, Ytest))
	print('Elapsed time for batch SGD with lr exponential decay:', dt4)

	# plot the cost
	#plt.plot(LL_exp)
	#plt.title('Cost for batch SGD with lr exponential decay')
	#plt.show()

'''
	print('\nBatch SGD with constant learning rate:')
	print('final error rate:', final_err1)
	print('elapsed time:', dt1)

	print('\nBatch SGD with RMSProp:')
	print('final error rate:', final_err2)
	print('elapsed time:', dt2)

	print('\nBatch SGD with AdaGrad:')
	print('final error rate:', final_err3)
	print('elapsed time:', dt3)



	# plot the costs together:   
	plt.plot(LL_batch, label='const_lr')
	plt.plot(LL_RMSProp, label='RMSProp')
	plt.plot(LL_AdaGrad, label='AdaGrad')
	#plt.plot(LL_exp, label='lr_exp_decay')
	plt.legend()
	plt.show()




if __name__ == '__main__':
	main()