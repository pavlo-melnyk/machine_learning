import numpy as np 
import pandas as pd 



def init_weights_and_biases(M1, M2):
	W = np.random.randn(M1, M2) / np.sqrt(M1/2.0)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)


def get_mnist_data(normalize=False):
	'''Returns data X matrix (Nx784) and label Y matrix (Nx1).
	'''
	# data filepath:
	filepath = '.../mnist_train.csv'
	# load the data:
	df = pd.read_csv(filepath)
	data = df.as_matrix()
	X, Y = data[:, 1:].astype(np.float32), data[:, 0]

	if normalize:
		# mean = X.mean(axis=0)
		# std = X.std(axis=0)
		# # np.place(std, std==0, 1)
		# X = (X - mean) / std
		X /= 255.0

	return X, Y