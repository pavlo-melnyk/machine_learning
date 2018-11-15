import numpy as np 
import matplotlib.pyplot as plt 

from datetime import datetime


# generate the data:
x = np.linspace(0, 3*np.pi, 10000)
N = len(x)
y = np.sin(x) + np.random.randn(N) + 5
# y = 2*np.random.randn(300) + np.sin(np.linspace(0, 3*np.pi, 300))

# visualize the data:
plt.plot(x, y)
plt.title('y = sin(x) + noise')
plt.show()


beta = 0.999

avg = np.zeros(N)
avg_bc = np.zeros(N) # for constant decay
avg_1 = np.zeros(N) # for alpha(t) = 1/t
t0 = datetime.now()

# calculate the exp smoothed averages:
avg[0] = (1 - beta) * y[0]
avg_1[0] = y[0] # for the first moment of time

# bias correction:
avg_bc[0] = avg[0] / (1 - beta)
for t in range(1, N):
	alpha = 1/(t+1) # t+1 because indexing starts with 0, s.t. avg_1[0] corresponds to t=1 and so on
	avg_1[t] = (1 - alpha) * avg_1[t-1] + alpha * y[t]

	# for the constant decay:
	avg[t] = beta * avg[t-1] + (1 - beta) * y[t]
	# bias correction:
	avg_bc[t] = avg[t] / (1 - beta**(t+1))

#dt = datetime.now() - t0

#print('np.mean(y):', np.mean(y))

# visualize:
plt.plot(x, y, label='original sin(x) with noise')
# plt.plot(x, (np.sin(x) + 5) , label='sin(x)')
plt.plot(x, np.ones(N)* np.mean(y), label='true_mean')
plt.plot(x, avg_1, label='exp weighted avg', lw=2.5)
plt.title('alpha(t) = 1/t')
plt.legend()
plt.show()


plt.plot(x, y, label='original sin(x) with noise')
# plt.plot(x, (np.sin(x) + 5) , label='sin(x)')
plt.plot(x, np.ones(N)* np.mean(y), label='true_mean')
plt.plot(x, avg, label='exp weighted avg', lw=2.5)
plt.plot(x, avg_bc, label='exp weighted avg with bias correction', lw=1.5)
plt.title('alpha(t) = %.3f' % (1-beta))
plt.legend()
plt.show()

#print('Elapsed time:', dt)