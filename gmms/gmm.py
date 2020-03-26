import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mv_normal



def gmm(X, K, n_iter=20, smoothing=1e-7, thre=0.1, visualize=False):
	np.random.seed(1)

	N, D = X.shape

	R = np.zeros((N, K))      # the responsibilities

	# parameters to learn: 
	M = np.zeros((K, D))      # the means
	C = np.zeros((K, D, D))   # DxD cov matrix for each Gaussian
	PI = np.ones(K) / K       # p(Z)=priors init to be uniform
	
	for k in range(K):
		M[k] = X[np.random.choice(N)]
		C[k] = np.eye(D)      # np.diag(np.ones(D)) spherical with variance 1


	################# Expectation Maximization #################
	costs = np.zeros(n_iter)
	weighted_pdfs = np.zeros((N, K)) # for storing PDF values

	for i in range(n_iter):
		##### Step 1: calculate the responsibilities #####
		# 	for k in range(K):
		# 		for n in range(N):
		# 			weighted_pdfs[n,k] = PI[k]*mv_normal.pdf(X[n], M[k], C[k])

		# 	# calculate the responsibilites using weighted_pdfs:
		# 	for k in range(K):
		# 		for n in range(N):
		# 			R[n,k] = weighted_pdfs[n,k] / weighted_pdfs[n,:].sum()

		# vectorized:
		for k in range(K):
			weighted_pdfs[:, k] = PI[k]*mv_normal.pdf(X, M[k], C[k])
		R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

		###### Step 2: (re)calculate the model parameters #####
		for k in range(K):
			Nk = R[:, k].sum()
			PI[k] = Nk / N
			M[k] = R[:, k].T @ X / Nk    # (1, N) @ (N, D) = (1, D)
			C[k] =  R[:, k] * (X - M[k]).T @ (X - M[k]) / Nk # (D, D)

			# hack to avoid C[k] being singular:
			C[k] += np.eye(D)*smoothing

		# update the cost - compute the log-likelihood:
		# J = sum_over_n( log( sum_over_k( PI[k] * p(X[n] | M[k], C[k]) ) ) )
		costs[i] = np.log(weighted_pdfs.sum(axis=1)).sum()

		# exit as early as possible:
		if i > 0:
			# check if converged:
			if np.abs(costs[i] - costs[i-1]) < thre:
				break

	if visualize:
		plt.figure(1)
		plt.plot(costs)
		plt.title('log-likelihood\nn_iter = '+str(i))
		# plt.show()

		plt.figure(2)
		colors =  R @ np.random.random((K, 3))
		plt.scatter(X[:, 0], X[:, 1], c=colors)
		plot_ellipses(M, C)
		plt.title('Learned Clusters')
		plt.show()

	print('\npi\'s:\n', PI)
	print('\nmeans:\n', M)
	print('\ncovariances:\n', C)

	return R



def plot_ellipses(M, C, lw=0.5, color='r'):
	''' Inspired by Per-Erik Forssén's em_demo.ipynb's
	plot_ellipse function (see the last item at 
	http://users.isy.liu.se/cvl/perfo/software/ for details)
	'''

	for k in range(M.shape[0]):
		d, E = np.linalg.eig(C[k])
		Rs = 2.0 * E @ np.diag(np.sqrt(d)) # radiuses

		phi = np.linspace(0.0, 2.0*np.pi, 150, endpoint=True)
		
		xy = np.array([np.cos(phi), np.sin(phi)])
		xy_m = Rs @ xy
		
		plt.scatter(M[k, 0], M[k, 1], c=color, s=16)
		plt.plot(xy_m[0] + M[k, 0], xy_m[1] + M[k, 1], c=color, linewidth=lw)

	

def main():
	np.random.seed(2)

	# generate some data:
	N = 2000 # n_samples
	D = 2   # n_features (will be H*W*D in the project)
	s = 4   #
	mu1 = np.array([0, 0]) # the means - centroids
	mu2 = np.array([s, s])
	mu3 = np.array([0, s])

	X = np.zeros((N, D))
	X[:1200, :] = 2*np.random.randn(1200, D) + mu1
	X[1200:1800, :] = np.random.randn(600, D) + mu2
	X[1800:, :] = 0.5*np.random.randn(200, D) + mu3

	# visualize:
	plt.scatter(X[:, 0], X[:, 1])
	plt.title('data')
	plt.show()

	# set number of modes (clusters):
	K = 3

	gmm(X, K, n_iter=100, thre=1e-3, visualize=True)
	


if __name__ == '__main__':
	main()