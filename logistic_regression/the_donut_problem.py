import numpy as np 
import matplotlib.pyplot as plt 
from Cocentric_circles_dataset import dataset as donuts

# get the data
X = donuts[:,:2]
T = donuts[:,2]

N, D = X.shape

# plot the data:
plt.scatter(X[:, 0], X[:, 1], c=T , marker='o')
plt.axis('equal')
plt.show()

# add a bias column:

# Let's get the radiuses of each point: 
# r^2 = x^2 + y^2   =>   r = (x^2 + y^2)^1/2

r = np.zeros((N,1))

for i in range(N):
    r[i] = np.sqrt(X[i,:].dot(X[i,:]))

# and add them to the dataset
Xb = np.hstack((r, np.ones((N,1)), X))



# Logistic Regression part:

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(T, Y):
	return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum() 


w = np.random.randn(D+2) / np.sqrt(D+2)
learning_rate = 0.0001
l1 = 1
errors = []

# Gradient Descent:
for i in range(5000):
	Y = sigmoid(Xb.dot(w))
	w = w - learning_rate*(Xb.T.dot(Y-T) + l1*np.sign(w))
	error = cross_entropy(T, Y) + l1*np.abs(w).sum()
	errors.append(error)
	if i % 100 == 0:
		print('i: %d, error: %f' %(i, error))

plt.plot(errors)
plt.title('Cross-entropy')
plt.show()

print('Final weights:', w)
print('Final classification rate:', (T==np.round(Y)).mean())