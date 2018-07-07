import numpy as np 
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 
from sklearn.utils import shuffle


def load_data():
	X, T = [], []
	for line in open('data_2d.csv'):
		line = line.split(',')
		X.append([float(x) for x in line[:-1]])
		T.append(float(line[-1]))
	X, T = np.array(X), np.array(T)
	# add a bias column
	X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1) 
	X, T = shuffle(X, T) # shuffle the data
	return X, T


def get_dataset(X, T, set_type='training', show=False):
	N, D = X.shape

	trn_idx = int(N*0.6)	            # the last index for training set
	cv_idx = trn_idx + int(N*0.2)		# the last index for cv-set
	if set_type == 'training':
		# return 60% of the shuffled data
		X, T = X[:trn_idx,:], T[:trn_idx]
		if show:
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			ax.scatter(X[:,1], X[:,2], T)
			plt.title(set_type + ' set')
			plt.show()		
		return X, T
	elif set_type == 'cv':
		# return other 20% of the shuffled data
		X, T = X[trn_idx:cv_idx,:], T[trn_idx:cv_idx]
		if show:
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			ax.scatter(X[:,1], X[:,2], T)
			plt.title(set_type + ' set')
			plt.show()		
		return X, T
	elif set_type == 'test':
		# return other 20% of the shuffled data
		X, T = X[cv_idx:,:], T[cv_idx:]
		if show:
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			ax.scatter(X[:,1], X[:,2], T)
			plt.title(set_type + ' set')
			plt.show()		
		return X, T


class LinearRegression():
	
	def predict(self, X):
		return X.dot(self.w)

	def fit(self, X, T, learning_rate=0.0001, max_i=1000, l1=2, l2=1, plt_cost=True):		
		# get the data shape:
		N, D = X.shape

		# randomly initialize weights:		
		self.w = np.random.randn(D) / np.sqrt(D)
		costs = []
		# Gradient Descent:
		for i in range(max_i):
			# get the prediction
			Y = self.predict(X)
			# update the weights:		
			self.w = self.w - learning_rate * (X.T.dot(Y - T) + l1*np.sign(self.w) + l2*self.w) / N		
			# calculate cost:
			mse = self.cost(X, T, l1, l2)
			costs.append(mse)
			print('iteration: %d,  cost = %f' % (i, mse))
		# plot the cost, if required
		if plt_cost:			
			plt.plot(costs, label='cost')
			plt.legend()
			plt.show()
		print('r_sq = ', self.get_r_sq(X, T))

	def cost(self, X, T, l1=0, l2=0):
		Y = self.predict(X)
		delta = T - Y
		return (delta.dot(delta) + l1*np.abs(self.w).sum() + l2*self.w.dot(self.w.T)) / len(T) 

	def get_r_sq(self, X, T):
		Y = self.predict(X)
		d1 = T - Y
		d2 = T - np.mean(T)
		return 1 - d1.dot(d1) / d2.dot(d2)


def main():
	# load data:
	X, T = load_data()	
	# train the model:
	model = LinearRegression()
	Xtrain, Ttrain = get_dataset(X, T, 'training', show=True)
	model.fit(Xtrain, Ttrain)
	print('optimized weights:', model.w)

	# evaluate on test set:
	Xtest, Ttest = get_dataset(X, T,'test')
	print('cost for the test set:', model.cost(Xtest, Ttest))

	prediction = model.predict(Xtest)

	# visualize in 3D:
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(Xtest[:,1], Xtest[:,2], Ttest, label='labels')
	ax.scatter(Xtest[:,1], Xtest[:,2], prediction, color='r', label='predicted')
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_zlabel('y')
	plt.title('test data')
	plt.legend()
	plt.show()
	
	# visualize targets in 2D:
	plt.plot()
	plt.plot(Ttest, label='targets')
	plt.plot(model.predict(Xtest), label='prediction', color='g')
	plt.legend()
	plt.show()



if __name__ == '__main__':
	main()



