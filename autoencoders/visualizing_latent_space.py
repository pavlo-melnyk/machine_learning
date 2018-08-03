import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from utils import get_mnist_data
from variational_autoencoder_tf import VariationalAutoencoder


def main():
	X, Y = get_mnist_data(normalize=True)
	# binarize the pictures:
	# X = (X > 0.5).astype(np.float32)
	
	N, D = X.shape

	# display some images:
	for i in range(len(X)):
		x = X[i]
		plt.imshow(x.reshape(28, 28), cmap='gray')
		plt.title('Image of \'%d\'' % Y[i])
		plt.show()

		prompt = input('Continue displaying?\n')
		if prompt and prompt[0] in ['n', 'N']:
			break

	# in order to visualize the latent space,
	# we'll make it of dimensionality 2, so
	# the last layer of the ENCODER will have just 2 units:
	vae = VariationalAutoencoder(784, [200, 100, 2])

	# the fit() method of the VariationalAutoencoder class
	# performs mini-batch SGD - it shuffles the input data;
	# you can make sure of it:

	# vae.fit(X)
	# for i in range(len(X)):
	# 	x = X[i]
	# 	plt.imshow(x.reshape(28, 28), cmap='gray')
	# 	plt.title('Image of \'%d\'' % Y[i])
	# 	plt.show()
	# 	prompt = input('Continue displaying?\n')
	# 	if prompt and prompt[0] in ['n', 'N']:
	# 		exit()

	# for later visualization we need to retain the original order,
	# to do this, we will pass in a copy of X:
	vae.fit(X.copy(), epochs=15)

	# get the latent space:
	Z = vae.transform(X) # returns 2 values for each sample - calculated means

	# make a scatter plot:
	plt.scatter(Z[:, 0], Z[:, 1], c=Y, s=15, alpha=0.5)
	plt.title('Visualization of the Latent Space')
	plt.colorbar()
	plt.show()


	# plot what image is reproduced for different parts of Z:

	# we are going to use the prior_predictive_given_input_probs() method;
	# first, we generate an empty image to fill it in later:
	n = 25 # number of images per side
	image = np.empty((28 * n, 28 * n)) # D is the height and width of the input samples
	
	# generate the latent vectors:
	Z = [] # a list of latern vectors
	x_values = np.linspace(-3, 3, n) 
	y_values = np.linspace(-3, 3, n)
	# we are going to make a prediction for all input latent vectors simultaneously:
	for x in x_values:
		for y in y_values:
			z = [x, y]
			Z.append(z)

	# reconstructed images:
	X_reconstructed = vae.prior_predictive_given_input_probs(Z)

	idx = 0
	for i, x in enumerate(x_values):
		for j, y in enumerate(y_values):
			x_reconstructed = X_reconstructed[idx].reshape(28, 28)
			idx += 1
			# map it to a particular part of the pre-created image 
			image[i*28 : (i+1)*28, j*28 : (j+1)*28] = x_reconstructed
	
	plt.imshow(image, cmap='gray')
	plt.show()


		

if __name__ == '__main__':
	main()






