import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

from scipy.stats import multivariate_normal as mvn 


class BayesClassifier:
	def fit(self, X, Y):
		# number of classes = number of unique elements of Y:
		self.K = len(set(Y))
		self.N = len(X)

		# gaussians for every class:
		self.gauss = []

		# the probability of class, p(Y), for every class:
		self.p_y = []

		# assuming that classes are in [0, K-1], 
		# calculate stats for every class:
		for i in range(self.K):
			Xi = X[Y==i]
			# calculate the mean per 'column':
			mean_Xi = np.mean(Xi, axis=0) 
			# calculate the covariance matrix (should be DxD, so we transpose Xi):
			cov_Xi = np.cov(Xi.T)
			# save to the storage:
			self.gauss.append({'mean': mean_Xi, 'cov': cov_Xi})
			# the probability of class, p(Y=k) = #k_class_samples / #all_samples:
			self.p_y.append(len(Xi)/self.N)

	def sample(self, y):
		mean = self.gauss[y]['mean']
		cov = self.gauss[y]['cov']
		return mvn.rvs(mean=mean, cov=cov)

	def random_sample(self):
		y = np.random.choice(self.K, p=self.p_y)
		return self.sample(y)


def main():
	# data filepath:
	filepath = '.../mnist_train.csv'

	# load the data:
	df = pd.read_csv(filepath)
	data = df.as_matrix()

	X, Y = data[:, 1:], data[:, 0]

	# classifier:
	cl = BayesClassifier()
	cl.fit(X, Y)

	for i in range(cl.K):
		plt.subplot(1, 2, 1)
		plt.imshow(cl.sample(i).reshape(28, 28), cmap='gray')
		plt.title('Generated Sample')
		plt.subplot(1, 2, 2)
		plt.imshow(cl.gauss[i]['mean'].reshape(28, 28), cmap='gray')
		plt.title('Mean')
		plt.suptitle('class "%s"'%i)
		plt.show()

if __name__ == '__main__':
	main()