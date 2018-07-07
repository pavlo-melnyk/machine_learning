import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(z):
	return 1/(1+np.exp(-z))


def cross_entropy(T, Y):
	J = -(T*np.log(Y)+(1-T)*np.log(1-Y)).mean()
	return J


# generate the data - two Gaussian clouds:
N = 100
D = 2 

X = np.random.randn(N, D)

# Let first 50 points be centered at X1 = -2, X2 = -2
X[:50, :] = X[:50, :] - 2*np.ones((50,D))

# and the second 50 points be centered at X1 = +2, X2 = +2
X[50:, :] = X[50:, :] + 2*np.ones((50,D))

# the targets:
T = np.array([0]*50 + [1]*50)

# add a column of ones for the bias term:
Xb = np.hstack((np.ones((N, 1)), X))

# random weights' initialization:
w = np.random.randn(D + 1) / np.sqrt(D + 1)

# logistic regression output (predicted):
Y = sigmoid(Xb.dot(w))
print(cross_entropy(T, Y))

# let's apply the closed-formed solution:
w = np.array([0, 4, 4])

# let's visualize our dataset:
plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 1000)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()

Y = sigmoid(Xb.dot(w))
print(cross_entropy(T, Y))

# Gradient Descent:
learning_rate = 0.1
lmb = 1 # lambda - l2-regularization parameter
for i in range(1000):
	if i % 10 == 0:
			print(cross_entropy(T, Y))

	w = w - learning_rate * (Xb.T.dot(Y - T) + lmb * w)
	Y = sigmoid(Xb.dot(w))

print('Final weights:', w)

# let's visualize our solution:
plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 1000)
y_axis = -x_axis
plt.plot(x_axis, y_axis, label='closed-form solution')
y_axis_reg = -(w[1]*x_axis + w[0]) / w[2]
plt.plot(x_axis, y_axis_reg, label='regularized solution')
plt.legend()
plt.show()
