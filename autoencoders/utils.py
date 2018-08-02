import numpy as np 
import pandas as pd 
import os


def init_weights_and_biases(M1, M2):
	W = np.random.randn(M1, M2) / np.sqrt(M1/2.0)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)


def get_mnist_data(normalize=False, filepath='train.csv'):
	'''Returns data X matrix (Nx784) and label Y matrix (Nx1).
	'''
	
	if not os.path.exists(filepath):
		print('\nPlease download the data from https://www.kaggle.com/c/digit-recognizer')
		print('and place \'train.csv\' in the current working directory.')
		print('Also, you can place it whenever you want and specify the corresponding filepath')
		print('as the \'filepath\' argument of this function')
		exit()

	print('processing the data.....')
	
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


def get_fashion_mnist_data(normalize=False, filepath1='fashion-mnist_train.csv', filepath2='fashion-mnist_test.csv'):
	'''Returns the tuple (X, Y, Xtest, Ytest)'''
	labels = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	if not os.path.exists(filepath):
		print('\nPlease download the data from https://www.kaggle.com/zalando-research/fashionmnist')
		print('and place \'fashion-mnist_train.csv\' and \'fashion-mnist_test.csv\' in the current working directory.')
		print('Also, you can place it whenever you want and specify the corresponding filepaths')
		print('as the \'filepath1\' and \'filepath2\' arguments of this function')
		exit()


	print('processing train set.....')	
	df1 = pd.read_csv(filepath1)
	train_data = df1.as_matrix().astype(np.float32)
	X, Y = train_data[:, 1:], train_data[:, 0]
	del df1

	print('processing test set.....')	
	df2 = pd.read_csv(filepath2)	
	test_data = df2.as_matrix().astype(np.float32)
	Xtest, Ytest = test_data[:, 1:], test_data[:, 0]
	del df2

	# shuffle the data:
	X, Y = shuffle(X, Y) 
	Xtest, Ytest = shuffle(Xtest, Ytest)
	
	if normalize:
		X /= 255.0
		Xtest /= 255.0