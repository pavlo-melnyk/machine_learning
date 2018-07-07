import numpy as np 
import matplotlib.pyplot as plt 


activation = 'swish'

def forward(X, W1, b1, W2, b2):
	
	Z = X.dot(W1) + b1

	if activation == 'swish':
		Z = Z * 1 / (1 + np.exp(-Z))

	elif activation == 'relu':
		# rectifier linear unit:
		Z[Z < 0] = 0

	elif activation == 'sigmoid':
		# sigmoid:
		Z = 1 / (1 + np.exp(-Z))

	A = Z.dot(W2) + b2

	# softmax:
	expA = np.exp(A)
	Y = expA / np.sum(expA, axis=1, keepdims=True)

	return Y, Z


def derivative_W2(Z, T, Y):
	return Z.T.dot(Y - T)


def derivative_b2(T, Y):
	return (Y - T).sum(axis=0)


def derivative_W1(X, Z, T, Y, W2):
	if activation == 'swish':
		sigmoid = 1 / (1 + np.exp(-Z))
		dZ = sigmoid + Z * sigmoid*(1 - sigmoid*sigmoid) * 10 
		return X.T.dot((Y - T).dot(W2.T)*dZ)

	elif activation == 'relu':      
		return X.T.dot((Y - T).dot(W2.T)*(Z > 0))

	elif activation == 'sigmoid':
		return X.T.dot((Y - T).dot(W2.T)*Z*(1 - Z))

	 
def derivative_b1(Z, T, Y, W2):
	if activation == 'swish':
		sigmoid = 1 / (1 + np.exp(-Z))
		dZ = sigmoid + Z * sigmoid*(1 - sigmoid*sigmoid) * 10
		return ((Y - T).dot(W2.T)*dZ).sum(axis=0)
		
	elif activation == 'relu':
		return ((Y - T).dot(W2.T)*(Z > 0)).sum(axis=0)

	elif activation == 'sigmoid':
		return ((Y - T).dot(W2.T)*Z*(1 - Z)).sum(axis=0)