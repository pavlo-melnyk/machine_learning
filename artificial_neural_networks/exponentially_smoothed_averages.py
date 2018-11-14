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
plt.title('y = sin(x)')
plt.show()


betta = 0.999
# betta = 0
avg = np.zeros(N)
avg_bc = np.zeros(N)
t0 = datetime.now()

# calculate the exp smoothed averages:
avg[0] = (1 - betta) * y[0]
# bias correction:
avg_bc[0] = avg[0] / (1 - betta)
for t in range(1, N):
	# betta = (1 - 1/(t)) 
	avg[t] = betta * avg[t-1] + (1 - betta) * y[t]
	# bias correction:
	avg_bc[t] = avg[t] / (1 - betta**(t+1))

#dt = datetime.now() - t0

#print('np.mean(y):', np.mean(y))

# visualize:
plt.plot(x, y, label='original sin(x) with noise')
# plt.plot(x, (np.sin(x) + 5) , label='sin(x)')
plt.plot(x, np.ones(N)* np.mean(y), label='true_mean')
plt.plot(x, avg, label='exp weighted avg', lw=2.5)
plt.plot(x, avg_bc, label='exp weighted avg with bias correction', lw=1.5)
plt.title('decay = %.3f' % betta)
plt.legend()
plt.show()

#print('Elapsed time:', dt)