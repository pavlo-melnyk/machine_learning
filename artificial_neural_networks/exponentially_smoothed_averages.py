import numpy as np 
import matplotlib.pyplot as plt 

from datetime import datetime


# generate the data:
x = np.linspace(0, 2*np.pi, 500)
N = len(x)
y = np.sin(x) + np.random.randn(N) + 5

# visualize the data:
plt.plot(x, y)
plt.title('y = sin(x)')
plt.show()


betta = 0.9
avg = np.zeros(N)
avg_bc = np.zeros(N)
t0 = datetime.now()

# calculate the exp smoothed averages:
for t in range(N-1):
	if t == 0:
		# bias correction:
		avg_bc[t] = (1 - betta) * y[0] / (1 - betta)
	avg[t+1] = betta * avg[t] + (1 - betta) * y[t+1]
	avg_bc[t+1] = avg[t+1] / (1 - betta**(t+1))

dt = datetime.now() - t0


# visualize:
plt.plot(x, y, label='original sin(x) with noise')
#plt.plot(x, (np.sin(x)+5) , label='sin(x)')
plt.plot(x, avg, label='exp weighted avg', lw=2.5)
plt.plot(x, avg_bc, label='exp weighted avg with bias correction', lw=1.5)
plt.title('decay = %.3f' % betta)
plt.legend()
plt.show()

#print('Elapsed time:', dt)