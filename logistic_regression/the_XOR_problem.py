import numpy as np 
import matplotlib.pyplot as plt 
from XOR_dataset import dataset 

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(T, Y):
	return -(T*np.log(Y) + (1-T)*np.log(1-Y)).mean() 


# make the values in range(0, 1) (originally, the range is (-1, 1))
X = (dataset[:,:2] + 1) / 2
T = dataset[:,2]

N, D = X.shape

# visualize the data
plt.scatter(X[:,0], X[:,1], c=T)
plt.show()

# a column of ones for the bias term:
ones = np.ones((N,1))

# create another feature to add to the dataset:
xy = np.matrix(X[:,0]*X[:,1]).T

# add the column of ones an the new feature to the dataset:
Xb = np.array(np.hstack((ones, xy, X)))

w = np.random.randn(D+2) / np.sqrt(D+2)
learning_rate = 0.001
errors = []

# Gradient Descent:
for i in range(50000):
	Y = sigmoid(Xb.dot(w))
	w = w - learning_rate*(Xb.T.dot(Y-T))
	error = cross_entropy(T, Y)
	errors.append(error)
	if i % 1000 == 0:
		print('i: %d, error: %f' % (i, error))

plt.plot(errors)
plt.title('Cross-entropy')
plt.show()

print('Final weights:', w)

print('Final classification rate:', (T==np.round(Y)).mean())