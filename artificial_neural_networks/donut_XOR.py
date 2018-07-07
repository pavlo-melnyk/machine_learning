import numpy as np 
import matplotlib.pyplot as plt 
 
from XOR_dataset import dataset as XOR_data
from Cocentric_circles_dataset import dataset as donuts_data
from sklearn.utils import shuffle
from datetime import datetime


def forward(X, W1, b1, W2, b2):
	Z = sigmoid(X.dot(W1) + b1)
	Y = sigmoid(Z.dot(W2) + b2) # since we are doing binary classification
	return Y, Z

def predict(X, W1, b1, W2, b2):
	Y, _ = forward(X, W1, b1, W2, b2)
	return np.round(Y)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def classification_rate(T, Y):
	return (T==Y).mean()


def derivative_W2(Z, T, Y):	
	return Z.T.dot(T - Y)


def derivative_b2(T, Y):
	# returns (1xK) bias vector:
	return (T - Y).sum()


def derivative_W1(X, Z, T, Y, W2):
	N, D = X.shape
	M = len(W2)
	K = 1
	ret2 = X.T.dot((T - Y).dot(W2.T)*Z*(1 - Z))

	return ret2


def derivative_b1(T, Y, W2, Z):
	# returns (1xM) bias vector:
	return ((T - Y).dot(W2.T)*Z*(1 - Z)).sum(axis=0)


def cost(T, Y):
	# binary cross-entropy:
	return -np.sum(T*np.log(Y) + (1 - T)*np.log(1 - Y))


def XOR_problem():
	# get the data for the XOR-problem:
	X = XOR_data[:-100,:-1]
	Y = XOR_data[:-100,-1].astype(np.int32)
	Xtest = XOR_data[-100:,:-1]
	
	# visualize the data:
	plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
	plt.show()

	# number of samples and features:
	N, D = X.shape	
	# hidden layer size (number of hidden neurons):
	M = 4	
	# number of classes:
	K = len(set(Y))

	Y = Y.reshape(N, 1)
	
	# randomly initialize the weights:
	# DxM matrix of weights for input-hidden layer connection:
	W1 = np.random.randn(D, M)
	# 1xM bias vector for the hidden layer:
	b1 = np.random.randn(M)
	# MxK matrix of weights for hidden-output layer connection:
	W2 = np.random.randn(M, 1)
	# 1xK bias vector for the output layer:
	b2 = np.random.randn(1)

	learning_rate = 0.0005
	costs = []
	t0 = datetime.now()

	for epoch in range(100000):
		output, hidden = forward(X, W1, b1, W2, b2)

		if epoch % 1000 == 0:
			c = cost(Y, output)
			P = predict(X, W1, b1, W2, b2)
			r = classification_rate(Y, P)
			print('Cost:', c, ' Classification rate:', r)
			costs.append(c)

		# Gradient Ascent:
		W2 += learning_rate * derivative_W2(hidden, Y, output)
		b2 += learning_rate * derivative_b2(Y, output)
		W1 += learning_rate * derivative_W1(X, hidden, Y, output, W2)
		b1 += learning_rate * derivative_b1(Y, output, W2, hidden)

	time_delta = datetime.now() - t0
	
	plt.plot(costs)
	plt.show()
	print('Elapsed time:', time_delta)
	plt.scatter(Xtest[:,0], Xtest[:,1], c=predict(Xtest, W1, b1, W2, b2))
	plt.plot(np.linspace(0, 1, 10), 0.5*np.ones(10), color='b')
	plt.plot(0.5*np.ones(10), np.linspace(0, 1, 10), color='b')
	plt.title('Predicted classes for test set')
	plt.show()


def donuts_problem():
	# get the data for the donut-problem:
	data = donuts_data
	X = data[:,:-1]
	Y = data[:,-1].astype(np.int32)
	X, Y = shuffle(X, Y)
	Xtest, Ytest = X[-200:, :], Y[-200:]      # test set
	X, Y = X[:-200,:], Y[:-200]               # train set
	
	
	# visualize the data:
	plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
	plt.axis('equal')
	plt.title('Donuts')
	plt.show()

	# number of samples and features:
	N, D = X.shape	
	# hidden layer size (number of hidden neurons):
	M = 4	
	# number of classes:
	K = len(set(Y))

	Y = Y.reshape(N, 1)
	
	# randomly initialize the weights:
	# DxM matrix of weights for input-hidden layer connection:
	W1 = np.random.randn(D, M)
	# 1xM bias vector for the hidden layer:
	b1 = np.random.randn(M)
	# MxK matrix of weights for hidden-output layer connection:
	W2 = np.random.randn(M, 1)
	# 1xK bias vector for the output layer:
	b2 = np.random.randn(1)

	learning_rate = 0.0005
	lmbd = 0.01 # L2-regularization
	costs = []
	t0 = datetime.now()

	for epoch in range(100000):
		output, hidden = forward(X, W1, b1, W2, b2)

		if epoch % 1000 == 0:
			c = cost(Y, output)
			P = predict(X, W1, b1, W2, b2)
			r = classification_rate(Y, P)
			print('Cost:', c, ' Classification rate:', r)
			costs.append(c)

		# Gradient Ascent:
		W2 += learning_rate * (derivative_W2(hidden, Y, output) - lmbd * W2) 
		b2 += learning_rate * (derivative_b2(Y, output) - lmbd * b2)
		W1 += learning_rate * (derivative_W1(X, hidden, Y, output, W2) - lmbd * W1)
		b1 += learning_rate * (derivative_b1(Y, output, W2, hidden) - lmbd * b1)

	time_delta = datetime.now() - t0
	
	plt.plot(costs)
	plt.title("Cross-entropy for the train set")
	plt.show()
	print('Elapsed time:', time_delta)
	
	# Evaluate on the test set:
	predY = predict(Xtest, W1, b1, W2, b2)
	print('\n\nClassification rate for the test set:',\
	 classification_rate(Ytest.reshape(len(Xtest), 1), predY))
	plt.scatter(Xtest[:,0], Xtest[:,1], c=predY)
	plt.title('Predicted classes for test set')
	plt.axis('equal')
	plt.show()



if __name__ == '__main__':
	XOR_problem()
'''
if __name__ == '__main__':
	donuts_problem()
'''
