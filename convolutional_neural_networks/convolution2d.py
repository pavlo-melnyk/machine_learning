import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from scipy.signal import convolve2d
from datetime import datetime


# def conv2d(img, W, mode='', visualize=False):
# 	n1, m1 = img.shape
# 	n2, m2 = W.shape
# 	print('Image shape:', img.shape, ' Filter shape:', W.shape)

# 	out_n = n1 + (n2 - 1)
# 	out_m = m1 + (m2 - 1)

# 	if visualize:
# 		plt.subplot(1, 2, 1)
# 		plt.imshow(img, cmap='gray')
# 		plt.title('the Image')

# 		plt.subplot(1, 2, 2)
# 		plt.imshow(W, cmap='gray')
# 		plt.title('the Filter')

# 		plt.show()

# 	out = np.zeros((out_n, out_m))
# 	for i in range(n1):
# 		for j in range(m1):
# 			for ii in range(n2):
# 				for jj in range(m2):
# 					# to make it not go outside the borders [0:n1; 0:m1]:
# 					# if i >= ii and j >= jj and i - ii < n1 and j - jj < m1:
# 					out[i, j] += W[ii, jj] * img[i-ii, j-jj]

# 	if mode=='same':
# 		out = out[:n1, :m1]
# 		print('Output shape:', out.shape)
# 		return out

# 	else:
# 		print(out.shape)
# 		return out				


def conv2d(img, W, mode='', visualize=False):
	n1, m1 = img.shape
	n2, m2 = W.shape
	print('Image shape:', img.shape, ' Filter shape:', W.shape)

	out_n = n1 + (n2 - 1)
	out_m = m1 + (m2 - 1)

	if visualize:
		plt.subplot(1, 2, 1)
		plt.imshow(img, cmap='gray')
		plt.title('the Image')

		plt.subplot(1, 2, 2)
		plt.imshow(W, cmap='gray')
		plt.title('the Filter')

		plt.show()

	out = np.zeros((out_n, out_m))
	for i in range(n1):
		for j in range(m1):
			out[i : (i+n2), j : (j+m2)] += W * img[i, j]

	if mode == 'same':
		out = out[n2//2:n1+n2//2, m2//2:m1+m2//2]
		print('Output shape:', out.shape)
		return out

	if mode == 'smaller':
		out=out[n2-1:n1, m2-1:m1]
		print('Output shape:', out.shape)
		return out

	else:
		print(out.shape)
		return out				



def main():
	# load the famous lena image:
	img = mpimg.imread('lena.png')

	# make it B&W:
	bw = img.mean(axis=2)

	# define the Sobel filter for vertical edges detection:
	Hx = np.array([
		[1, 0, -1],
		[2, 0, -2],
		[1, 0, -1]
	], dtype=np.float32)

	# define Gaussian filter:
	W = np.zeros((20, 20))
	for i in range(20):
		for j in range(20):
			dist = (i - 9.5)**2 + (j - 9.5)**2
			W[i, j] = np.exp(-dist / 50)
	W /= W.sum() # normalize the filter
	
	t0 = datetime.now()

	# convlove it with the filter:	
	out = conv2d(bw, W, mode='same', visualize=True)

	dt = datetime.now() - t0

	plt.imshow(out, cmap='gray')
	plt.show()

	print('Elapsed time:', dt)


if __name__ == '__main__':
	main()