import os 
import numpy as np

from sklearn.utils import shuffle
from scipy.misc import imread, imsave, imresize
from glob import glob



def init_weights_and_biases(M1, M2):
	W = np.random.randn(M1, M2) / np.sqrt(M1/2.0)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)
	

def conv_cond_concat(x, y):
	import tensorflow as tf
	"""Concatenate conditioning vector on feature map axis."""
	x_shapes = x.get_shape()
	# print('x_shapes:', x_shapes)
	y_shapes = y.get_shape()
	# print('y_shapes:', y_shapes)
	return tf.concat(
			[x, y*tf.ones([y_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 
			axis=3
		)


def y2indicator(y):
	N = len(y)
	K = len(set(y))
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, y[i].astype(np.int32)] = 1
	return ind


def scale_image(im):
	# scale to (-1, +1)
	return (im / 255.0)*2 - 1


def files2images(filenames):
	return [scale_image(imread(fn)) for fn in filenames]


def get_bob_ross_data(dim=196):
	print('flag')
	path = '.../bob_ross_challenge/'
	if not os.path.exists(path+'data'):
		print('\n".../bob_ross_challenge/data" not found\nedit the path and add data')
		exit()
	
	# eventual place where our final data will reside:
	if not os.path.exists(path+'reshaped_data'):
		# load in the original images:
		filenames = glob(path+'data/*.png'+'data/*.jpg')
		N = len(filenames)
		print('Found %d files!' % N)

		# reshape the data:
		os.mkdir(path+'reshaped_data')
		
		for i in range(N):
			# crop_and_resave(filenames[i], path+'align_and_cropped', offset_h=108)
			im = imread(filenames[i])
			small = imresize(im, (dim, dim))

			filename = filenames[i].split('/')[-1].split('\\')[-1]
			imsave('%s/%s' % (path+'reshaped_data', filename), small)
			
			if i % 100 == 0:
				print('%d/%d' % (i, N))

	# make sure to return the cropped version:
	filenames = glob(path+'reshaped_data/*.png'+'reshaped_data/*.jpg')
	return filenames