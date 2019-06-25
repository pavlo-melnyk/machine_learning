import os
import sys

if 'cpu' in sys.argv:
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt 
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Dropout, Flatten
from keras.datasets import mnist



class CustomDense(Dense):
	def __init__(self, units, freeze_ratio=0.1, seed=None, verbose=False, visualize=False, **kwargs):
		self.units = units
		self.freeze_ratio = freeze_ratio
		self.seed = seed # to reproduce the results
		self.verbose = verbose
		self.visualize = visualize
		super(CustomDense, self).__init__(units, **kwargs)


	def build(self, input_shape):
		kernel_shape = (input_shape[-1], self.units)
		
		n_weights = np.prod(kernel_shape)
		weights = []

		n_non_trainable_weights = int(n_weights*self.freeze_ratio)
		chunk_size = max(1, int(self.freeze_ratio*kernel_shape[1]))
		n_chunks = n_non_trainable_weights//chunk_size

		if self.verbose:
			#check the dimensions:	
			print('\nlayer:', self.name)							
			print('\ninput_shape:', input_shape)
			print('kernel_shape:', kernel_shape)
			print('\nfrozen weights:', n_non_trainable_weights)
			print('# of chunks (tensors):', n_chunks)
			print('chunk_size:', chunk_size, end='\n\n')

		for i in range(n_chunks):
			weights.append(
				self.add_weight(name='W_frozen',
								shape=(chunk_size, ),
								initializer=self.kernel_initializer,
								regularizer=self.kernel_regularizer, 
								trainable=False)
			)

		if (n_non_trainable_weights % n_chunks != 0):
			weights.append(
				self.add_weight(name='W_frozen',
								shape=(n_non_trainable_weights % chunk_size, ),
								initializer=self.kernel_initializer,
								regularizer=self.kernel_regularizer, 
								trainable=False)
			)	

		assert K.eval(K.concatenate(weights)).shape[0] == n_non_trainable_weights
	
		n_trainable_weights = n_weights - n_non_trainable_weights		
		chunk_size = max(1, kernel_shape[0]//10) # # of weights per tensor
		n_chunks = n_trainable_weights//chunk_size # # of tensors

		if self.verbose:
			print('\ntrainable weights:', n_trainable_weights)
			print('# of chunks (tensors):', n_chunks)
			print('chunk_size:', chunk_size, end='\n\n\n')

		for i in range(n_chunks):
			weights.append(
				self.add_weight(name='W',
								shape=(chunk_size, ),
								initializer=self.kernel_initializer,
								regularizer=self.kernel_regularizer, 
								trainable=True)
			)

		if (n_trainable_weights % n_chunks != 0):
			weights.append(
				self.add_weight(name='W',
								shape=(n_trainable_weights % chunk_size, ),
								initializer=self.kernel_initializer,
								regularizer=self.kernel_regularizer, 
								trainable=True)
			)	
		
		assert K.eval(K.concatenate(weights)).shape[0] == n_weights

		np.random.seed(self.seed)
		np.random.shuffle(weights)
		
		self.kernel = K.reshape(K.concatenate(weights), kernel_shape)
		
		if self.visualize:
			# we need to know what weights are trainable:
			kernel = np.empty(0)
			for i, variable in enumerate(weights):
				if 'frozen' in variable.name:
					kernel = np.concatenate([kernel, np.zeros(variable.shape)])
				else:
					kernel = np.concatenate([kernel, np.ones(variable.shape)])
			kernel = np.reshape(kernel, kernel_shape)
			
			plt.pcolor(kernel, edgecolors='k', linewidths=0.1)
			plt.axes().set_aspect('equal')
			plt.title(str(kernel_shape[0]) + ' x ' + str(kernel_shape[1]) + ' weight matrix')
			plt.show()
			del kernel			

		del weights
		
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.units,),
										initializer=self.bias_initializer,
										name='bias',
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
	
		self.built = True
		# super(CustomDense, self).build(input_shape) # don't call it, throws an exception


	def call(self, inputs):
		output = K.dot(inputs, self.kernel)
		if self.use_bias:
			output = K.bias_add(output, self.bias, data_format='channels_last')
		if self.activation is not None:
			output = self.activation(output)
		return output



def get_mnist_data(reshape=False, normalize=False):
	# load the MNIST data:
	(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()	
	Xtrain = np.expand_dims(Xtrain, axis=3)
	Xtest = np.expand_dims(Xtest, axis=3)
	
	if reshape:
		N, K = len(Ytrain), len(set(Ytrain))
		D = Xtrain.shape[1]*Xtrain.shape[2]
		# reshape the data to be (NxD):
		Xtrain, Xtest = Xtrain.reshape(N, D), Xtest.reshape(len(Xtest), D)

	if normalize:
		Xtrain = np.float32(Xtrain / 255.0)
		Xtest = np.float32(Xtest / 255.0)

	return (Xtrain, Ytrain), (Xtest, Ytest)



def main():
	freeze_ratio = 0.01
	seed = 1996
	visualize = True
	verbose = True

	# load normalized data:
	(Xtrain, Ytrain), (Xtest, Ytest) = get_mnist_data(reshape=True, normalize=True)

	# check the data:
	i = np.random.choice(len(Xtrain))
	plt.imshow(Xtrain[i].reshape(28, 28), cmap='gray')
	plt.title(Ytrain[i])
	plt.show()

	N, D = Xtrain.shape # training data dimesionality
	K = len(set(Ytrain)) # number of classes

	model = Sequential()	
	model.add(CustomDense(256, freeze_ratio=freeze_ratio, seed=seed, verbose=verbose,
						visualize=visualize, input_shape=(D,), activation='relu'))	
	model.add(CustomDense(K, freeze_ratio=freeze_ratio, seed=seed, verbose=verbose,
						visualize=visualize, activation='softmax'))	
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	
	r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=10)

	# plot the losses:
	plt.plot(r.history['loss'], label='loss')
	plt.plot(r.history['val_loss'], label='val_loss')
	plt.legend()
	plt.show()

	# and the accuracies:
	plt.plot(r.history['acc'], label='acc')
	plt.plot(r.history['val_acc'], label='val_acc')
	plt.legend()
	plt.show()



if __name__ == '__main__':
	main()