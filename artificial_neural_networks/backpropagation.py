import numpy as np 
import matplotlib.pyplot as plt 

from datetime import datetime


def softmax(A):
	return np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
	Z = sigmoid(X.dot(W1) + b1)
	Y = softmax(Z.dot(W2) + b2)
	return Y, Z


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def classification_rate(T, Y):
	return (T==Y).mean()


def derivative_W2(Z, T, Y):
	N, K = T.shape
	M = Z.shape[1]
	'''
	# slow:
	ret1 = np.zeros((M, K))
	for n in range(N):
		for m in range(M):
			for k in range(K):
				ret1[m,k] += (T[n,k] - Y[n,k])*Z[n,m]
	'''
	# fast:
	ret2 = Z.T.dot(T - Y)
	
	return ret2


def derivative_b2(T, Y):
	# returns (1xK) bias vector:
	return (T - Y).sum(axis=0)


def derivative_W1(X, Z, T, Y, W2):
	N, D = X.shape
	M, K = W2.shape
	'''
	# slow:
	ret1 = np.zeros((D, M))
	for n in range(N):
		for k in range(K):
			for m in range(M):
				for d in range(D):
					ret1[d,m] += (T[n,k] - Y[n,k])*W2[m,k]*Z[n,m]*(1 - Z[n,m])*X[n,d] 
	'''
	# fast:
	ret2 = X.T.dot((T - Y).dot(W2.T)*Z*(1 - Z))

	return ret2


def derivative_b1(T, Y, W2, Z):
	# returns (1xM) bias vector:
	return ((T - Y).dot(W2.T)*Z*(1 - Z)).sum(axis=0)


def cost(T, Y):
	# log-likelihood:
	return np.sum(T*np.log(Y))


def main():
	# number of features:
	D = 2
	# hidden layer size (number of hidden neurons):
	M = 3
	# number of classes:
	K = 3
	# number of samples per each class:
	Nclass = 500

	# generate the data:
	X1 = np.random.randn(Nclass, 2) + np.array([0, -2]) # a Gaussian centred in (0,-2)
	X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
	X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
	X = np.vstack((X1, X2, X3))
	Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

	N = len(Y)

	T = np.zeros((N, K))
	# one hot encoding for the targets:
	for i in range(N):
		T[i, Y[i]] = 1

	# visualize the data:
	plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
	plt.show()

	# randomly initialize the weights:
	# DxM matrix of weights for input-hidden layer connection:
	W1 = np.random.randn(D, M)
	# 1xM bias vector for the hidden layer:
	b1 = np.random.randn(1, M)
	# MxK matrix of weights for hidden-output layer connection:
	W2 = np.random.randn(M, K)
	# 1xK bias vector for the output layer:
	b2 = np.random.randn(K)

	learning_rate = 10e-7
	costs = []
	t0 = datetime.now()

	for epoch in range(100000):
		output, hidden = forward(X, W1, b1, W2, b2)
		if epoch % 1000 == 0:
			c = cost(T, output)
			P = np.argmax(output, axis=1)
			r = classification_rate(Y, P)
			print('Cost:', c, ' Classification rate:', r)
			costs.append(c)

		# Gradient Ascent:
		W2 += learning_rate * derivative_W2(hidden, T, output)
		b2 += learning_rate * derivative_b2(T, output)
		W1 += learning_rate * derivative_W1(X, hidden, T, output, W2)
		b1 += learning_rate * derivative_b1(T, output, W2, hidden)

	time_delta = datetime.now() - t0
	plt.plot(costs)
	plt.show()
	print('Elapsed time:', time_delta)


if __name__ == '__main__':
	main()

