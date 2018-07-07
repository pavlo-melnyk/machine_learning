import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from sklearn.utils import shuffle 
from datetime import datetime 

from util import get_transformed_data, forward, error_rate, cost, grad_W, grad_b, y2indicator


def main():
	X, Y, _, _ = get_transformed_data()
	X = X[:, :300]
	
	# normalize the data:
	mu = X.mean(axis=0)
	std = X.std(axis=0)
	X = (X - mu) / std

	print('Performing logistic regression...')
	Xtrain, Ytrain = X[:-1000, :], Y[:-1000]
	Xtest, Ytest = X[-1000:, :], Y[-1000:]

	N, D = Xtrain.shape
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)
	K = len(set(Y)) 

	np.random.seed()


	# 1. Full Gradient Descend:
	W = np.random.randn(D, K) / np.sqrt(D)
	b = np.zeros(K)
	LL = [] # a storage for costs
	lr = 0.0001 # learning rate
	reg = 0.01 # L2-regularization term
	t0 = datetime.now()
	print('utilizing full GD...')
	for i in range(200):
		p_y = forward(Xtrain, W, b)	

		W += lr*(grad_W(Ytrain_ind, p_y, Xtrain) - reg*W)
		b += lr*(grad_b(Ytrain_ind, p_y).sum(axis=0) - reg*b)

		p_y_test = forward(Xtest, W, b)
		ll = cost(p_y_test, Ytest_ind)
		LL.append(ll)
			
		if i%10 == 0:
			error = error_rate(p_y_test, Ytest)
			print('i: %d, cost: %.6f, error: %.6f' % (i, ll, error))
	dt1 = datetime.now() - t0
	p_y_test = forward(Xtest, W, b)
	plt.plot(LL)
	plt.title('Cost for full GD')
	plt.show()
	plt.savefig('Cost_full_GD.png')
	print('Final error rate:', error_rate(p_y_test, Ytest))
	print('Elapsed time for full GD:', dt1)
	

	# 2. Stochastic Gradien Descent
	W = np.random.randn(D, K) / np.sqrt(D)
	b = np.zeros(K)
	LLstochastic = [] # a storage for costs
	lr = 0.0001 # learning rate
	reg = 0.01 # L2-regularization term
	t0 = datetime.now()
	print('utilizing stochastic GD...')
	for i in range(25):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		# we consider just 500 samples, not all the dataset
		for n in range(N):
			x = tmpX[n, :].reshape(1, D)
			y = tmpY[n, :].reshape(1, K)
			p_y = forward(x, W, b)

			W += lr*(grad_W(y, p_y, x) - reg*W)
			b += lr*(grad_b(y, p_y).sum(axis=0) - reg*b)

			p_y_test = forward(Xtest, W, b)
			ll = cost(p_y_test, Ytest_ind)
			LLstochastic.append(ll)

			if n % (N//2) == 0:
				error = error_rate(p_y_test, Ytest)
				print('i: %d, cost: %.6f, error: %.6f' % (i, ll, error))
	
	dt2 = datetime.now() - t0
	p_y_test = forward(Xtest, W, b)
	plt.plot(LLstochastic)
	plt.title('Cost for stochastic GD')
	plt.show()
	plt.savefig('Cost_stochastic_GD.png')
	print('Final error rate:', error_rate(p_y_test, Ytest))
	print('Elapsed time for stochastic GD:', dt2)
	

	# 3. Batch Gradient Descent:
	W = np.random.randn(D, K) / np.sqrt(D)
	b = np.zeros(K)
	LLbatch = []
	lr = 0.0001 # learning rate
	reg = 0.01 # L2-regularization term
	batch_size = 500
	n_batches = N // batch_size
	t0 = datetime.now()
	print('utilizing batch GD...')
	for i in range(50):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			x = tmpX[j*batch_size:batch_size*(j+1),:]
			y = tmpY[j*batch_size:batch_size*(j+1),:]
			p_y = forward(x, W, b)

			W += lr*(grad_W(y, p_y, x) - reg*W)
			b += lr*(grad_b(y, p_y).sum(axis=0) - reg*b)

			p_y_test = forward(Xtest, W, b)
			ll = cost(p_y_test, Ytest_ind)
			LLbatch.append(ll)

			if j % (n_batches//2) == 0:
				error = error_rate(p_y_test, Ytest)
				print('i: %d, cost: %.6f, error: %.6f' % (i, ll, error))
	dt3 = datetime.now() - t0
	p_y_test = forward(Xtest, W, b)
	plt.plot(LLbatch)
	plt.title('Cost for batch GD')
	plt.show()
	plt.savefig('Cost_batch_GD.png')
	print('Final error rate:', error_rate(p_y_test, Ytest))
	print('Elapsed time for batch GD', dt3)


	# plot all costs together:
	x1 = np.linspace(0, 1, len(LL))
	plt.plot(x1, LL, label='full')

	x2 = np.linspace(0, 1, len(LLstochastic))
	plt.plot(x2, LLstochastic, label='stochastic')

	x3 = np.linspace(0, 1, len(LLbatch))
	plt.plot(x3, LLbatch, label='batch')

	plt.legend()
	plt.show()
	plt.savefig('Costs_together.png')
	

if __name__ == '__main__':
	main()