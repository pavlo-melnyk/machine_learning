import numpy as np 
import matplotlib.pyplot as plt 

from mlp import forward, derivative_W2, derivative_b2, derivative_W1, derivative_b1
from util import get_normalized_data, cost, y2indicator, error_rate
from sklearn.utils import shuffle
from datetime import datetime 


def main():
	# compare 2 scenarios:	
	# 1. batch GD with RMSProp and momentum
	# 2. Adam GD

	max_iter = 20 
	print_period = 10

	X, Y = get_normalized_data()
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
	W1_0 = np.random.randn(D, M) / np.sqrt(D)
	b1_0 = np.zeros(M)
	W2_0 = np.random.randn(M, K) / np.sqrt(M)
	b2_0 = np.zeros(K)

	
	# 1. batch GD with RMSProp and momentum:	
	print('\nperforming batch GD with RMSProp and momentum...')
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()	

	LL_rm = []
	CR_rm = []

	# hyperparams:
	lr0 = 0.001
	#lr0 = 0.0001
	mu = 0.9
	decay = 0.999
	eps = 10e-9

	# momentum (velocity terms):
	dW1 = 0
	db1 = 0
	dW2 = 0
	db2 = 0

	# rms-prop cache (with no bias correction):
	cache_W2 = 1
	cache_b2 = 1
	cache_W1 = 1
	cache_b1 = 1


	t0 = datetime.now()
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_size:(j+1)*batch_size, :]
			Ybatch = Ytrain_ind[j*batch_size:(j+1)*batch_size, :]
			p_Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)
			#print(Z.shape, p_Ybatch.shape, Ybatch.shape)
			#print('First batch cost:', cost(p_Ybatch, Ybatch))

			# updates:
			# (note: we utilize a bit different version of momentum)
			gW2 = (derivative_W2(Z, Ybatch, p_Ybatch) + reg*W2)
			cache_W2 = decay*cache_W2 + (1 - decay)*gW2*gW2
			dW2 = mu*dW2 + (1 - mu)*lr0*gW2 / (np.sqrt(cache_W2 + eps))
			W2 -= dW2
			#dW2 = mu*dW2 - lr0*gW2 / (np.sqrt(cache_W2) + eps)
			#W2 += dW2

			gb2 = (derivative_b2(Ybatch, p_Ybatch) + reg*b2)
			cache_b2 = decay*cache_b2 + (1 - decay)*gb2*gb2
			db2 = mu*db2 + (1 - mu)*lr0*gb2 / (np.sqrt(cache_b2 + eps))
			b2 -= db2
			#db2 = mu*db2 - lr0*gb2 / (np.sqrt(cache_b2) + eps)
			#b2 += db2
			
			gW1 = (derivative_W1(Xbatch, Z, Ybatch, p_Ybatch, W2) + reg*W1)
			cache_W1 = decay*cache_W1 + (1 - decay)*gW1*gW1
			dW1 = mu*dW1 + (1 - mu)*lr0*gW1 / (np.sqrt(cache_W1 + eps))
			W1 -= dW1
			#dW1 = mu*dW1 - lr0*gW1 / (np.sqrt(cache_W1) + eps)
			#W1 += dW1
			
			gb1 = (derivative_b1(Z, Ybatch, p_Ybatch, W2) + reg*b1)
			cache_b1 = decay*cache_b1 + (1 - decay)*gb1*gb1
			db1 = mu*db1 + (1 - mu)*lr0*gb1 / (np.sqrt(cache_b1 + eps))
			b1 -= db1
			#db1 = mu*db1 - lr0*gb1 / (np.sqrt(cache_b1) + eps)
			#b1 += db1

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				#print('pY:', pY)
				ll = cost(pY, Ytest_ind)
				LL_rm.append(ll)
				print('\ni: %d, j: %d, cost: %.6f' % (i, j, ll))

				error = error_rate(pY, Ytest)
				CR_rm.append(error)
				print('error rate:', error)

	dt1 = datetime.now() - t0
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('\nFinal error rate:', error_rate(pY, Ytest))
	print('Elapsed time for batch GD with RMSProp and momentum:', dt1)

	# plot the cost
	plt.plot(LL_rm)
	plt.title('Cost for batch GD with RMSProp and momentum')
	plt.show()



	# 2. Adam optimizer
	print('\nperforming Adam optimizer...')
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()	

	# hyperparams:
	lr = 0.001
	beta1 = 0.9
	beta2 = 0.999
	eps = 10e-9

	# 1st moment:
	mW1 = 0
	mb1 = 0
	mW2 = 0
	mb2 = 0

	# 2nd moment:
	vW1 = 0
	vb1 = 0
	vW2 = 0
	vb2 = 0

	LL_adam = []
	CR_adam = []
	t0 = datetime.now()
	t = 1 # index; used instead of j, because j starts with 0
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_size:(j+1)*batch_size,:]
			Ybatch = Ytrain_ind[j*batch_size:(j+1)*batch_size,:]
			p_Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# updates:
			# gradients:
			gW2 = derivative_W2(Z, Ybatch, p_Ybatch) + reg*W2
			gb2 = derivative_b2(Ybatch, p_Ybatch) + reg*b2
			gW1 = derivative_W1(Xbatch, Z, Ybatch, p_Ybatch, W2) + reg*W1
			gb1 = derivative_b1(Z, Ybatch, p_Ybatch, W2) + reg*b1

			# 1st moment:
			mW2 = beta1*mW2 + (1 - beta1)*gW2
			mb2 = beta1*mb2 + (1 - beta1)*gb2
			mW1 = beta1*mW1 + (1 - beta1)*gW1
			mb1 = beta1*mb1 + (1 - beta1)*gb1
	
			# 2nd moment:
			vW2 = beta2*vW2 + (1 - beta2)*gW2*gW2
			vb2 = beta2*vb2 + (1 - beta2)*gb2*gb2
			vW1 = beta2*vW1 + (1 - beta2)*gW1*gW1
			vb1 = beta2*vb1 + (1 - beta2)*gb1*gb1

			# bias correction:
			mW2_bc = mW2 / (1 - beta1**t)
			mb2_bc = mb2 / (1 - beta1**t)
			mW1_bc = mW1 / (1 - beta1**t)
			mb1_bc = mb1 / (1 - beta1**t)

			vW2_bc = vW2 / (1 - beta2**t)
			vb2_bc = vb2 / (1 - beta2**t)
			vW1_bc = vW1 / (1 - beta2**t)
			vb1_bc = vb1 / (1 - beta2**t)

			# weights and biases (parameters):
			W2 = W2 - lr*mW2_bc / np.sqrt(vW2_bc + eps) 
			b2 = b2 - lr*mb2_bc / np.sqrt(vb2_bc + eps) 
			W1 = W1 - lr*mW1_bc / np.sqrt(vW1_bc + eps)
			b1 = b1 - lr*mb1_bc / np.sqrt(vb1_bc + eps)

			t += 1

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_adam.append(ll)
				print('\ni: %d, j: %d, cost: %.6f' % (i, j, ll))

				error = error_rate(pY, Ytest)
				CR_adam.append(error)
				print('error rate:', error) 

	dt2 = datetime.now() - t0
	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print('\nFinal error rate:', error_rate(pY, Ytest))
	print('Elapsed time for Adam optimizer:', dt2)

	# plot the cost
	plt.plot(LL_adam)
	plt.title('Cost for Adam optimizer')
	plt.show()


	# plot costs from the two experiments together:
	plt.plot(LL_rm, label='RMSProp with momentum')
	plt.plot(LL_adam, label='Adam optimizer')
	plt.title('Cost')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()