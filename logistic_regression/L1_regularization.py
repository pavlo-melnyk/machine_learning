import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(T, Y):
	return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum() 

# generate a 'fat' dataset matrix
N = 50
D = 50

# uniformly distributed numbers in range(-5, 5)
X = 10 * np.random.random((N,D)) - 5 

# true weights: only the first three affect the output, 
#the task of L1 regularization is to detect it - to achieve sparsity
true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

# the targets:
T = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5)) 

costs = []
w = np.random.randn(D) / np.sqrt(D)
b = 0
learning_rate = 0.001
l1 = 2 # the l1 penalty term

# Gradient Descent:
for i in range(5000):
	Y = sigmoid(X.dot(w) + b)
	w = w - learning_rate*(X.T.dot(Y-T) + l1*np.sign(w))
	b -= learning_rate*((Y-T).sum() + l1*np.sign(b))
	cost = cross_entropy(T, Y) + l1*np.abs(w).sum()
	costs.append(cost)
	if i % 100 == 0:
		print('i: %d, cost: %f' %(i, cost))

plt.plot(costs)
plt.show()

plt.plot(true_w, label='true w')
plt.plot(w, label='w map')
plt.title('l1 = ' + str(l1))
plt.legend()
plt.show()

print('Final weights:', w)


