# Object Localization project from  
# https://www.udemy.com/course/advanced-computer-vision/
# Step 1: localize white boxes on black b/g using a pre-trained VGG

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

if tf.__version__[0] == '2':
	# for TF 2.0 and newer:
	from tensorflow.keras.applications import VGG16
	from tensorflow.keras.layers import Flatten, Dense
	from tensorflow.keras.models import Model 
	from tensorflow.keras.optimizers import Adam, SGD

else:
	import keras
	from keras.applications import VGG16
	from keras.layers import Flatten, Dense
	from keras.models import Model 
	from keras.optimizers import Adam, SGD

from matplotlib.patches import Rectangle 



def image_generator(batch_size=64, n_batches=10):
	# generate 100x100 b/w samples and targets:
	while True:
		for _ in range(n_batches):
			# create placeholders:
			X = np.zeros((batch_size, 100, 100, 3))
			Y = np.zeros((batch_size, 4))

			for i in range(batch_size):
				# create and store the boxes:
				row0 = np.random.randint(90)
				col0 = np.random.randint(90)
				row1 = np.random.randint(row0, 100) # row1 >= row0
				col1 = np.random.randint(col0, 100) # col1 >= col0
				
				# fill in the white rectangle:
				X[i, row0:row1, col0:col1, :] = 1
				
				# normalize the targets to be in range [0, 1]:
				Y[i, 0] = row0 / 100            # top-left corner y-coord
				Y[i, 1] = col0 / 100            # tor-left corner x-coord
				Y[i, 2] = (row1 - row0) / 100   # height
				Y[i, 3] = (col1 - col0) / 100   # width

			# yield a batch of samples and targets:
			yield X, Y



def make_and_plot_prediction(model, x=0):
	if not x:
		# generate a random image:
		x = np.zeros((100, 100, 3))
		row0 = np.random.randint(90)
		col0 = np.random.randint(90)
		row1 = np.random.randint(row0, 100)
		col1 = np.random.randint(col0, 100)
		x[row0:row1, col0:col1, :] = 1

		print('\n\ngenerated test sample box coords\nrow: {}, col: {}, height: {}, width: {}'.format(row0, col0, row1-row0, col1-col0))

	# predict bounding box using the pre-trained model:
	p = model.predict(np.expand_dims(x, 0))[0]

	# reverse the transformation into un-normalized form:
	p *= 100
	print('\nprediction\nrow: %.3f, col: %.3f, height: %.3f, width: %.3f' % (p[0], p[1], p[2], p[3]))

	plot_prediction(x, p)



def plot_prediction(x, p):
	# draw the box:
	fig, ax = plt.subplots(1)
	ax.imshow(x)

	# need to specify [col, row, width, height]
	rect = Rectangle(
		(p[1], p[0]),
		p[3], p[2], 
		linewidth=1, edgecolor='r', facecolor='none'
	)

	ax.add_patch(rect)
	plt.show()



def main():
	# choose the loss:
	# loss = 'mse'
	loss = 'binary_crossentropy'

	# load the pre-trained model:
	vgg = VGG16(
		input_shape=[100, 100, 3], 
		include_top=False, 
		weights='imagenet')

	# construct our own model with transfer learning:
	# flatten the VGG 3x3x512 output:
	x = Flatten()(vgg.output)
	# output 4 predictions for the box ((row, col), height, width):
	x = Dense(4, activation='sigmoid')(x)
	# instantiate and compile our model:
	model = Model(vgg.input, x)
	model.compile(loss=loss, optimizer=Adam(lr=1e-4))

	model.summary()

	# # sanity check - test the generator:
	# gen = image_generator(64)
	# X, Y = next(gen)
	# x, y = X[0], Y[0]
	# plt.figure(0)
	# plt.imshow(x, cmap='gray')
	# plt.title('row: {}, col: {}, height: {}, width: {}'.format(100*y[0], 100*y[1], 100*y[2], 100*y[3]))
	# plt.show()

	# pass the data generator to our model and train the model:
	model.fit_generator(
		image_generator(64, 100), 
		steps_per_epoch=100,
		epochs=5,
	)

	for _ in range(10):
		make_and_plot_prediction(model)




if __name__ == '__main__':
	main()