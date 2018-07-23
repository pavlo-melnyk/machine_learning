import numpy as np 
import matplotlib.pyplot as plt

from sklearn.mixture import BayesianGaussianMixture
from datetime import datetime
from utils import get_mnist_data


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
			t0 = datetime.now()
			Xi = X[Y==i]
			# calculate the mean per feature:
			mean_Xi = np.mean(Xi, axis=0)
			# create a GMM model:
			gmm = BayesianGaussianMixture(n_components=10) # n_components = max # clusters
			# fit the data to the gmm:
			print('Fitting GMM', i)
			gmm.fit(Xi)
			print('elapsed time:', datetime.now() - t0, '\n')
			# save to the storage:
			self.gauss.append({'model': gmm, 'mean': mean_Xi})
			# the probability of class, p(Y=k) = #k_class_samples / #all_samples:
			self.p_y.append(len(Xi)/self.N)


	def sample(self, y):
		gmm = self.gauss[y]['model']
		# sample() returns a tuple (sample, cluster_the_sample_came_from):
		sample, label = gmm.sample()
		mean = self.gauss[y]['mean']
		# or using a sklearn non-public parameter:
		# mean = gmm.means_[label]
		return sample.reshape(28, 28), mean.reshape(28, 28)

	def random_sample(self):
		y = np.random.choice(self.K, p=self.p_y)
		return self.sample(y)



def main():
	# load the data:
	X, Y = get_mnist_data(normalize=True)

	# classifier:
	cl = BayesClassifier()
	cl.fit(X, Y)

	for i in range(cl.K):
		sample, mean = cl.sample(i)
		plt.subplot(1, 2, 1)
		plt.imshow(sample, cmap='gray')
		plt.title('Generated Sample')
		plt.subplot(1, 2, 2)
		plt.imshow(mean, cmap='gray')
		plt.title('Mean')
		plt.suptitle('class "%s"'%i)
		plt.show()

	sample, _ = cl.random_sample()
	plt.imshow(sample, cmap='gray')
	plt.title('a drawn sample')
	plt.show()

if __name__ == '__main__':
	main()