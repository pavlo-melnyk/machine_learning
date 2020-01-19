import numpy as np 
import matplotlib.pyplot as plt 


# Let's generate some data:
N = 23 
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)
n_outliers = 3

# We are going to mannually make some outliers, 
# so set the last point to 10 bigger, etc.
Y[-n_outliers:] += 10

plt.scatter(X, Y)
plt.xlabel('size')
plt.ylabel('price')
plt.title('Product Price as a (Linear) Function of Size') 
plt.grid()
plt.show()

# add the bias term:
X = np.vstack((np.ones(N), X)).T

# first calculate max likelihood solution:
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Y_ml = X.dot(w_ml) # predicted values

# plot the solution (X without the bias column)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_ml, label='maximum likelihood line', color='r')
plt.xlabel('size')
plt.ylabel('price')
plt.legend()
plt.title('Product Price as a (Linear) Function of Size')
plt.grid() 
plt.show()

# then calculate max likelihood solution 
# for the same data but without outliers:
X_cut = X[:-n_outliers]
Y_cut = Y[:-n_outliers]

w_ml_2 = np.linalg.solve(X_cut.T.dot(X_cut), X_cut.T.dot(Y_cut))
Y_ml_2 = X.dot(w_ml_2)

plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_ml, label='maximum likelihood line', color='r')
plt.plot(X[:,1], Y_ml_2, label='main trend', color='g')
plt.xlabel('size')
plt.ylabel('price')
plt.legend()
plt.title('Product Price as a (Linear) Function of Size')
plt.grid() 
plt.show()

# Now let's apply L2-regularization (Ridge Regression):
lmd = 1000
w_reg = np.linalg.solve(((X.T.dot(X)) + lmd*np.eye(2)), X.T.dot(Y))
Y_reg = X.dot(w_reg)

plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_ml, label='maximum likelihood line', color='r')
plt.plot(X[:,1], Y_ml_2, label='main trend', color='g')
plt.plot(X[:,1], Y_reg, label='regularized prediction', color='b')
plt.xlabel('size')
plt.ylabel('price')
plt.legend()
plt.title('Product Price as a (Linear) Function of Size')
plt.grid() 
plt.show()