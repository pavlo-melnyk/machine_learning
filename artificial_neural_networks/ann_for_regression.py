import numpy as np 
import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D


def forward(X, W1, b1, W2, b2):
	Z = np.tanh(X.dot(W1) + b1)
	Y = Z.dot(W2) + b2
	return Y, Z


def derivative_W2(Z, T, Y):
	return Z.T.dot(T - Y)


def derivative_b2(T, Y):
	return (T - Y).sum()


def derivative_W1(X, Z, T, Y, W2):
	return X.T.dot(np.outer((T - Y), W2) * (1 - Z**2)) 


def derivative_b1(Z, T, Y, W2):
	return (np.outer((T - Y), W2) * (1 - Z**2)).sum(axis=0)


def cost(T, Y):
	return (T - Y).T.dot(T - Y)


def r_sq(T, Y):
	d1 = (T - Y)
	d2 = (T.mean() - Y)
	return 1 - (d1.T.dot(d1)) / (d2.T.dot(d2))


def main():
	# generate the data:
	N = 500
	X = np.random.random((N, 2))*4 - 2 # in range(-2, +2)
	T = X[:,0]*X[:,1] # targets

	# visualize it:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X[:,0], X[:,1], T)
	plt.show()

	# create a neural network:
	D = 2
	M = 4

	# randomly initialize weights: 
	W1 = np.random.randn(D, M) / np.sqrt(D)
	b1 = np.zeros(M)

	W2 = np.random.rand(M) / np.sqrt(M)
	b2 = 0

	# Gradient Ascent:
	learning_rate = 0.0001
	regularization = 0
	costs = []

	for i in range(10000):
		Y, Z = forward(X, W1, b1, W2, b2)
		

		W2 += learning_rate * (derivative_W2(Z, T, Y) - regularization * W2)
		b2 += learning_rate * (derivative_b2(T, Y) - regularization * b2)
		W1 += learning_rate * (derivative_W1(X, Z, T, Y, W2) - regularization * W1)
		b1 += learning_rate * (derivative_b1(Z, T, Y, W2) - regularization * b1)

		if i % 1000 == 0:
			c = cost(T, Y)
			costs.append(c)
			print('i:', i, ' cost:', c, ' r_sq:', r_sq(T, Y))

	plt.plot(costs)
	plt.title('Cost for train set')
	plt.show()

	# visualize the prediction for the data:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X[:,0], X[:,1], T)

	# surface plot:
	line = np.linspace(-2, 2, 20)
	xx, yy = np.meshgrid(line, line)
	Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
	Y, _ = forward(Xgrid, W1, b1, W2, b2)
	ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Y, linewidth=0.2, antialiased=True)
	plt.show()


	# plot the magnitude of residuals:
	Ygrid = Xgrid[:,0] * Xgrid[:, 1]
	R = np.abs(Ygrid - Y)

	plt.scatter(Xgrid[:,0], Xgrid[:,1], c=R) 
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], R, linewidth=0.2, antialiased=True)
	plt.show()


if __name__ == '__main__':
	main()




