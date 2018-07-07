import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from scipy.signal import convolve2d


img = mpimg.imread('lena.png')

# make it black&white:
bw = img.mean(axis=2)

# Sobel operator - approximate gradient in X direction:
Hx = np.array([
	[-1, 0, 1],
	[-2, 0, 2], 
	[-1, 0, 1]
], dtype=np.float32())

# Sobel operator - approximate gradient in Y direction:
# Hy = Hx.T
Hy = np.array([
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]
], dtype=np.float32())

Gx = convolve2d(bw, Hx) # G stands for gradient
Gy = convolve2d(bw, Hy)

plt.subplot(1, 2, 1)
plt.imshow(Gx, cmap='gray')
plt.title('Vertical edges detection')
plt.subplot(1, 2, 2)
plt.imshow(Gy, cmap='gray')
plt.title('Horizontal edges detection')
plt.show()

# We could think of Gx and Gy sort of as vectors, so
# Gradient magnitude:
G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
plt.title('Edge detection')
plt.show()

# Gradients' direction:
theta = np.arctan2(Gy, Gx)
plt.imshow(theta, cmap='gray')
plt.title('Gradients direction')
plt.show()
