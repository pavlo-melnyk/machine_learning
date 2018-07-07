import numpy as np 
import matplotlib.pyplot as plt 

# make up some data
N = 100
X = np.linspace(0, 6*np.pi, N)
Y = np.sin(X)

plt.plot(X, Y)
plt.show()

def make_poly(X, deg):
	n = len(X)
	data = [np.ones(n)]
	for d in range(deg):
		data.append(X**(d+1))
	return np.vstack(data).T

def fit(X, Y):
	return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

def fit_and_display(X, Y, sample, deg):
	# shows how different polynomial degree fit the original data
	N = len(X)
	train_idx = np.random.choice(N, sample)
	Xtrain = X[train_idx]
	Ytrain = Y[train_idx]

	plt.scatter(Xtrain, Ytrain)
	plt.title('Original Training Set')
	plt.axis('equal')
	plt.show()

	# fit polynomial
	Xtrain_poly = make_poly(Xtrain, deg)
	w = fit(Xtrain_poly, Ytrain)

	# display the polynomial
	X_poly = make_poly(X, deg)
	Y_ = X_poly.dot(w)
	plt.plot(X, Y, label='original data')
	plt.plot(X, Y_, label='predicted')
	plt.scatter(Xtrain, Ytrain, label='training samples')
	plt.title('deg = %d' % deg)
	plt.legend()

	plt.show()

def get_mse(Y, Y_):
	# calculate the mean square error
	d = Y - Y_
	return d.dot(d) / len(d)


def plot_train_vs_test_curves(X, Y, sample=20, max_deg=20):
	# plots learning curves
	N = len(X)
	train_idx = np.random.choice(N, sample)
	Xtrain = X[train_idx]
	Ytrain = Y[train_idx]

	test_idx = [idx for idx in range(N) if idx not in train_idx]
	Xtest = X[test_idx]
	Ytest = Y[test_idx]

	mse_trains = []
	mse_tests = []
	for deg in range(max_deg+1):
		Xtrain_poly = make_poly(Xtrain, deg)
		w = fit(Xtrain_poly, Ytrain) # learn the weights from the training data
		Y_train = Xtrain_poly.dot(w) # predicted values for the training set
		mse_train = get_mse(Ytrain, Y_train)
		mse_trains.append(mse_train)

		Xtest_poly = make_poly(Xtest, deg)
		Y_test = Xtest_poly.dot(w) # predicted values for the test set
		mse_test = get_mse(Ytest, Y_test)
		mse_tests.append(mse_test)

	plt.plot(mse_trains, label='train mse')
	plt.plot(mse_tests, label='test mse')
	plt.xlabel('deg, polynomial degree')
	plt.ylabel('error')
	plt.legend()
	plt.show()

	plt.plot(mse_trains, label='train mse')
	plt.legend()
	plt.show()



for deg in (5, 6, 7, 8, 9):
	fit_and_display(X, Y, 10, deg)

plot_train_vs_test_curves(X, Y)