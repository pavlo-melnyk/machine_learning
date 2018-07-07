import numpy as np 
import matplotlib.pyplot as plt 

n = 1000

x = 2 * np.random.rand(n,2) - 1
labels = (x[:,0] * x[:,1] < 0)

dataset = np.column_stack((x, labels))

if __name__ == '__main__':
	plt.scatter(x[:,0], x[:, 1], c=labels)
	plt.show()

	print('Dataset size: %d x %d'%(dataset.shape))
