import numpy as np 
import matplotlib.pyplot as plt 


class KNN(object):
	def __init__(self, Xtrain, Ytrain, K=5):
		self.Xtrain = Xtrain
		self.Ytrain = Ytrain
		self.K = K
		
	def predict(self, X):
		'''
		Calculates the distance between an unseen sample x
		and every sample in Xtrain; 
		returns predicted class as the most frequent in K smallest
		distances
		'''
		Ntest, Dtest = X.shape
		Y = np.zeros((Ntest, 1))
		for i in range(Ntest):
			# calculate the distance vector (1xNtrain):
			distance = self.calculate_distance(X[i,:], self.Xtrain)
			# sort indexes in ascending order:
			idx = np.argsort(distance)
			# take just the K first (nearest) samples:
			idx = idx[:self.K]
			# add them to the neighbors storage:
			neighbors = self.Ytrain[idx]
			# choose the most frequent class:
			count = np.bincount(neighbors)
			if len(count) > self.K:
				Y[i] = count.argmax()
			else:
				Y[i] = neighbors[count.argmax()]
			#print('neighbors:', neighbors, '; most frequent:', Y[i])

		return Y.reshape(Ntest)


	def calculate_distance(self, x1, x2, dist_type='Euclidean'):
		if dist_type == 'Euclidean':
			return np.sqrt(np.sum((x1 - x2)**2, axis=1))


	def classification_rate(self, X, T):
		predY = self.predict(X)		
		return (T==predY).mean()