import numpy as np 
import matplotlib.pyplot as plt 

# fat matrix:
N = 50
D = 50

X = (np.random.random((N, D)) - 0.5)*10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3)) # last D-3 terms do not influence the output

Y = X.dot(true_w) + np.random.randn(N)*0.5

costs = []

# initialize weights:
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 100.0

for i in range(500):
	Y_ = X.dot(w)

	# Gradient Descent:	
	w = w - learning_rate * (X.T.dot(Y_ - Y) + l1*np.sign(w))

	mse = (Y - Y_).T.dot(Y - Y_) / N
	costs.append(mse)

plt.plot(costs)
plt.show()

print('final w: ', w)

plt.plot(true_w, label='true_w')
plt.plot(w, label='w_map')
plt.legend()
plt.show()