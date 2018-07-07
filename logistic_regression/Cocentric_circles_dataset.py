import numpy as np 
import matplotlib.pyplot as plt 

# number of points for each circle of a 'donut'
n = 500

# 1000 thetas in range [0; 360*]
theta1 = 2 * np.pi * np.random.random(n) 
theta2 = 2 * np.pi * np.random.random(n) 
# radius-vectors to generate points around the circle
r1 = 10 + 0.7*np.random.randn(n, 1)
r2 = 5 + 0.7*np.random.randn(n, 1)

# parametric circle equation:     x1(theta) = r*cos(theta); x2(theta) = r*sin(theta) 
x1 = (np.array([np.cos(theta1), np.sin(theta1)])).T
x2 = (np.array([np.cos(theta2), np.sin(theta2)])).T

# so we get element-wise multiplication of (n, 1) vector and (1000, 2) matrix
circle1 = r1 * x1 # remains (1000, 2) matrix
circle2 = r2 * x2 

# create a dataset for the circles:
# first n rows are two coordinates (x1, x2) of the circle1;
# next n rows are for the coordinates of the circle2
data = np.concatenate((circle1, circle2)) # (2*n, 2)
labels = np.concatenate((np.ones((n,1), dtype=np.int), np.zeros((n,1), dtype=np.int))) # (2*n, 1)
dataset = np.column_stack((data, labels)) # complete dataset

'''
# plot the 'donut'
plt.scatter(donuts[:, 0], donuts[:, 1], c=donuts[:, 2] , marker='o')
plt.axis('equal')
plt.show()

print(dataset[np.random.randint(0, 2*n, 100),])
print('Dataset size: %d x %d'%(dataset.shape))
input("\nPress 'Enter' to continue..................................")
'''