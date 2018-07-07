import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
#import theano.tensor as T

from theano_ann import ANN  
from sklearn.utils import shuffle
from datetime import datetime 


def grid_search():	
	# get the data:
	df = pd.read_csv('Spirals_dataset.csv')
	data = df.as_matrix()
	X = data[:,:-1]
	Y = data[:,-1]
	
	# visualize the data:
	plt.scatter(X[:,0], X[:,1], c=Y)
	plt.title('Spirals')
	plt.axis('equal')
	plt.show()

	# split the data:
	Xtrain, Ytrain = X[:-270, :], Y[:-270]
	Xtest, Ytest = X[-270:, :], Y[-270:]

	# hyperparameters to be tried:
	M = [
		[300],
		[100, 100],
		[50, 50, 50]
	]      # number of hidden layer neurons
	learning_rates = [1e-4, 1e-3, 1e-2]   # learning rate
	reg = [0., 1e-1, 1.0] # L2-regularization term
	
	best_validation_score = 0
	best_hl_size = None
	best_lr = None
	best_l2 = None
	t0 = datetime.now()
	# Grid Search loops:
	for hl_size in M:
		for lr in learning_rates:
			for l2 in reg:
				model = ANN(hl_size)
				model.fit(
					Xtrain,
					Ytrain, 
					learning_rate=lr, 
					reg=l2, 
					mu=0.9, 
					epochs=300, 
					show_fig=False
				)
				validation_score = model.score(Xtest, Ytest)
				train_score = model.score(Xtrain, Ytrain)
				print('\nvalidation set accuracy: %.3f,  training set accuracy: %.3f' % (validation_score, train_score))   
				print('hidden layer size: {}, learning rate: {}, l2: {}'.format(hl_size, lr, l2))

				if validation_score > best_validation_score:
					best_validation_score = validation_score
					best_hl_size = hl_size
					best_lr = lr
					best_l2 = l2
	dt = datetime.now() - t0
	print('\nElapsed time:', dt)
	print('\nBest validation accuracy:', best_validation_score)
	print('\nBest settings:')
	print('Best hidden layer size:', best_hl_size)
	print('Best learning rate:', best_lr)
	print('Best regularization term:', best_l2)
	print()



if __name__ == '__main__':
	grid_search()