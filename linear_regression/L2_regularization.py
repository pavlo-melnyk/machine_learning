import numpy as np 
import matplotlib.pyplot as plt 

# Let's generate some data:
N = 50 
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)

# We are going to mannually make some outliers, 
# so set the last point to 30 bigger, etc.
Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

# add the bias term:
X = np.vstack((np.ones(N), X)).T

# first calculate max likelihood solution:
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Y_ml = X.dot(w_ml) # predicted values

# plot the solution (X without the bias column)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_ml, label='maximum likelihood line', color='r')
plt.legend() 
plt.show()


# Now let's apply L2-regularisation (Ridge Regression):
lmd = 1000
w_reg = np.linalg.solve(((X.T.dot(X)) + lmd*np.eye(2)), X.T.dot(Y))
Y_reg = X.dot(w_reg)

plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_ml, label='maximum likelihood line', color='r')
plt.plot(X[:,1], Y_reg, label='regularised prediction', color='g')
plt.legend() 
plt.show()