import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

data_path = 'mnist_train.csv'

def get_transformed_data():
	print('Reading in and transforming data...')
	df = pd.read_csv(data_path)
	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)

	X = data[:, 1:]
	mu = X.mean(axis=0)
	X = X - mu 
	pca = PCA()
	Z = pca.fit_transform(X)
	Y = data[:, 0]

	return Z, Y, pca, mu


def get_normalized_data(show_img=False):
	print('Reading in and transforming data...')
	df = pd.read_csv(data_path)
	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)
	X = data[:, 1:]
	Y = data[:, 0]
	if show_img:
		n = np.random.randint(len(X))
		plt.imshow(255-X[n,:].reshape(28,28), cmap='gray')
		plt.title("Original image of '{}'".format(int(Y[n])))
		plt.show()
	mu = X.mean(axis=0)
	std = X.std(axis=0)
	np.place(std, std==0, 1) # replace entries of std=0 to 1
	X = (X - mu) / std
	if show_img:		
		plt.imshow(255-X[n,:].reshape(28,28), cmap='gray')
		plt.title("Normalized image of '{}'".format(int(Y[n])))
		plt.show()
	

	return X, Y


def plot_cumulative_variance(pca):
	P = []
	for p in pca.explained_variance_ratio_:
		if len(P) == 0:
			P.apppend(p)
		else:
			P.append(p + P[-1])
	plt.plot(P)
	plt.show()
	return P


def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 / 2.0)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def relu(x):
    return x * (x > 0)


def swish(a, tensorflow=False):
    if tensorflow:
        import tensorflow as tf
        return a * tf.sigmoid(a)
    else:
        return a / (1 + np.exp(-a))


def forward(X, W, b):
	# softmax:
	a = X.dot(W) + b
	#print('any nan in X?:', np.any(np.isnan(X)))
	#print('any nan in W?:'), np.any(np.isnan(W))
	#print('W:', W)
	#print('X.dot(W):', X.dot(W))
	#print('b:', b)
	#print('a:', a)
	expa = np.exp(a)
	#print('expa:', expa)
	y = expa / np.sum(expa, axis=1, keepdims=True)
	# exit()
	return y


def predict(p_y):
	return np.argmax(p_y, axis=1)


def error_rate(p_y, t):
	prediction = predict(p_y)
	return np.mean(prediction != t)


def cost(p_y, t):
	tot = t * np.log(p_y)
	return -tot.sum()


def grad_W(t, y, X):
	# return the gradient for W:
	return X.T.dot(t - y)	


def grad_b(t, y):
	return (t - y).sum(axis=0)


def y2indicator(y):
	N = len(y)
	y = y.astype(np.int32)
	K = len(set(y))
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, y[i]] = 1

	return ind


def benchmark_full():
	X, Y = get_normalized_data()

	print('Performing logistic regression...')
	#lr = LogisticRegression(solver='lbfgs')

	# test on the last 1000 points:
	#lr.fit(X[:-1000, :200], Y[:-1000]) # use only first 200 dimensions
	#print(lr.score(X[-1000:, :200], Y[-1000:]))
	#print('X:', X)

	# normalize X first
	#mu = X.mean(axis=0)
	#std = X.std(axis=0)
	#X = (X - mu) / std

	Xtrain = X[:-1000,]
	Ytrain = Y[:-1000]
	Xtest = X[-1000:,]
	Ytest = Y[-1000:]
	
	N, D = Xtrain.shape
	K = len(set(Ytrain))
	
	# convert Ytrain and Ytest to (N x K) matrices of indicator variables
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)

	# randomly initialize the weights and bias:
	W = np.random.randn(D, K) / np.sqrt(D)
	b = np.zeros(K)
	LL = []
	LLtest = []
	CRtest = []

	# reg = 1
	# learning rate 0.0001 is too high, 0.00005 is also too high:
	
	lr = 0.00004
	reg = 0.01
	for i in range(500):
		p_y = forward(Xtrain, W, b)
		# print('p_y:', p_y)
		ll = cost(p_y, Ytrain_ind)
		LL.append(ll)

		p_y_test = forward(Xtest, W, b)
		lltest = cost(p_y_test, Ytest_ind)
		LLtest.append(lltest)

		err = error_rate(p_y_test, Ytest)
		CRtest.append(err)

		W += lr*(grad_W(Ytrain_ind, p_y, Xtrain) - reg*W)
		b += lr*(grad_b(Ytrain_ind, p_y) - reg*b)
		if i % 10 == 0:
			print('Cost at iteration %d: %.6f' % (i, ll))
			print('Error rate:', err)

	p_y = forward(Xtest, W, b)
	print('Final error rate:', error_rate(p_y, Ytest)) 
	iters = range(len(LL))
	plt.plot(iters, LL, iters, LLtest)
	plt.show()
	plt.plot(CRtest)
	plt.show()


def benchmark_pca():
	X, Y, _, _ = get_transformed_data()
	X = X[:, :300]

	# normalize X first:
	mu = X.mean(axis=0)
	std = X.std(axis=0)
	X = (X - mu) / std

	print('Performing logistic regression...')
	Xtrain = X[:-1000,]
	Ytrain = Y[:-1000].astype(np.int32)
	Xtest = X[-1000:,]
	Ytest = Y[-1000:].astype(np.int32)

	N, D = Xtrain.shape
	K = len(set(Ytrain))
	Ytrain_ind = np.zeros((N, K))
	for i in range(N):
		Ytrain_ind[i, Ytrain[i]] = 1

	Ntest = len(Ytest)
	Ytest_ind = np.zeros((Ntest, K))
	for i in range(Ntest):
		Ytest_ind[i, Ytest[i]] = 1

	W = np.random.randn(D, K) / np.sqrt(D)
	b = np.zeros(10)
	LL = []
	LLtest = []
	CRtest = []

	# D = 300 -> error = 0.07
	lr = 0.0001
	reg = 0.01
	for i in range(200):
		p_y = forward(Xtrain, W, b)
		#print('p_y:', p_y)
		ll = cost(p_y, Ytrain_ind)
		LL.append(ll)

		p_y_test = forward(Xtest, W, b)
		lltest = cost(p_y_test, Ytest_ind)
		LLtest.append(lltest)

		err = error_rate(p_y_test, Ytest)
		CRtest.append(err)

		W += lr*(grad_W(Ytrain_ind, p_y, Xtrain) - reg*W)
		b += lr*(grad_b(Ytrain_ind, p_y) - reg*b)
		if i % 10 == 0:
			print('Cost at iteration %d: %.6f' % (i, ll))
			print('Error rate:', err)

	p_y = forward(Xtest, W, b)
	print('Final error rate:', error_rate(p_y, Ytest))
	iters = range(len(LL))
	plt.plot(iters, LL, iters, LLtest)
	plt.show()
	plt.plot(CRtest)
	plt.show()


if __name__ == '__main__':
	#benchmark_pca()
	benchmark_full()




