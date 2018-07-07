import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from scipy.signal import convolve2d


img = mpimg.imread('lena.png')

# plt.imshow(img)
# plt.show()

# make it black&white - take a mean along 3rd axis:
bw = img.mean(axis=2)

# plt.imshow(bw, cmap='gray')
# plt.show()

# define Gaussian filter:
W = np.zeros((20, 20))
for i in range(20):
	for j in range(20):
		dist = (i - 9.5)**2 + (j - 9.5)**2
		W[i, j] = np.exp(-dist / 50)
W /= W.sum() # normalize the filter

# plt.imshow(W, cmap='gray')
# plt.show()

# convolve bw with the filter:
out = convolve2d(bw, W)

# plt.imshow(out, cmap='gray')
# plt.show()

# print('Input shape:', bw.shape)
# print('Otput shape:', out.shape)

out = convolve2d(bw, W, mode='same')

# plt.imshow(out, cmap='gray')
# plt.show()

# print('Input shape:', bw.shape)
# print('Otput shape:', out.shape)

# convolve the original color image with the filter:
out3 = np.zeros(img.shape)
for i in range(3):
	out3[:, :, i] = convolve2d(img[:, :, i], W, mode='same')

# restrict the output after convolving to be in range (0, 1)
# if haven't normalized the filter:
# out3 /= out3.max() 
plt.imshow(out3)
plt.show()
