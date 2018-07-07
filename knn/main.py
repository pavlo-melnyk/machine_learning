import numpy as np 
import pandas as pd

from PCA import PCA
from KNN import KNN
from datetime import datetime


def get_data(pca_ON=False, print_shapes=False):
	data= pd.read_csv('mnist_train.csv').as_matrix()
	
	Xtrain = data[:-10000, 1:]
	Ytrain = data[:-10000, 0]
	Xtest = data[-10000:,1:]
	Ytest = data[-10000:,0]

	dataset = {}

	if pca_ON:
		pca = PCA(n_components=30)	 
		pca.fit(Xtrain)
		if print_shapes:
			print('\nEigenvectors size:', pca.evecs.shape)               
		Xtrain = pca.transform(Xtrain) 
		Xtest = pca.transform(Xtest) 

	if print_shapes:
		print('\nXtrain: {}, Ytrain: {}'.format(Xtrain.shape, Ytrain.shape))
		print('Xtest: {}, Ytest: {}'.format(Xtest.shape, Ytest.shape))
	

	dataset['train'] = (Xtrain, Ytrain)
	dataset['test'] = (Xtest, Ytest)
 			
	return dataset


def main():
	dataset = get_data(pca_ON=True, print_shapes=True)
	Xtrain, Ytrain = dataset['train']
	model = KNN(Xtrain, Ytrain, K=3)

	Xtest, Ytest = dataset['test']
	t0 = datetime.now()
	classification_rate = model.classification_rate(Xtest, Ytest)
	elapsed_time = datetime.now() - t0

	print('\nElapsed time:', elapsed_time)
	print('\n\nClassification rate with {} nearest neighbors: {}%\n\n'.\
		format(model.K, 100*np.round(classification_rate, 3)))
	

if __name__ == '__main__':
	main()
