# Object Localization project from  
# https://www.udemy.com/course/advanced-computer-vision/
# Step 2: localize an object of a fixed sized on black b/g

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

if tf.__version__[0] == '2':
	# for TF 2.0 and newer:
	from tensorflow.keras.applications import VGG16
	from tensorflow.keras.layers import Flatten, Dense
	from tensorflow.keras.models import Model 
	from tensorflow.keras.optimizers import Adam, SGD
	from tensorflow.keras.preprocessing import image

else:
	import keras
	from keras.applications import VGG16
	from keras.layers import Flatten, Dense
	from keras.models import Model 
	from keras.optimizers import Adam, SGD
	from keras.preprocessing import image

from imageio import imread

from matplotlib.patches import Rectangle 



IMG_DIM = 200



def make_model(loss='binary_crossentropy'):
	vgg = VGG16(
		input_shape=[IMG_DIM, IMG_DIM, 3],
		include_top=False,
		weights='imagenet')
	x = Flatten()(vgg.output)
	x = Dense(4, activation='sigmoid')(x)
	model = Model(vgg.input, x)
	model.compile(loss=loss, optimizer=Adam(lr=1e-4))	
	return model



def image_generator(ob_img, batch_size=64, n_batches=10):
	ob_H, ob_W = ob_img.shape[:2]
	# generate IMG_DIM x IMG_DIM samples and targets:
	while True:
		for _ in range(n_batches):
			# create placeholders:
			X = np.zeros((batch_size, IMG_DIM, IMG_DIM, 3))
			Y = np.zeros((batch_size, 4))

			for i in range(batch_size):
				# select a location for the object:
				row0 = np.random.randint(0, IMG_DIM - ob_H)
				col0 = np.random.randint(0, IMG_DIM - ob_W)
				row1 = row0 + ob_H # row1 >= row0
				col1 = col0 + ob_W # col1 >= col0
				
				# place the object:
				X[i, row0:row1, col0:col1, :] = ob_img
				
				# normalize the targets to be in range [0, 1]:
				Y[i, 0] = row0 / IMG_DIM            # top-left corner y-coord
				Y[i, 1] = col0 / IMG_DIM            # tor-left corner x-coord
				Y[i, 2] = (row1 - row0) / IMG_DIM   # height
				Y[i, 3] = (col1 - col0) / IMG_DIM   # width

			# yield a batch of samples and targets:
			yield X / 255., Y



def make_and_plot_prediction(model, x, y=''):
	if len(y) == 4:
		y *= IMG_DIM
		print('\n\ntarget\nrow: %d, col: %d, height: %d, width: %d' % (int(y[0]), int(y[1]), int(y[2]), int(y[3])))

	# predict bounding box using the pre-trained model:
	p = model.predict(np.expand_dims(x, axis=0))[0]

	# reverse the transformation into un-normalized form:
	p *= IMG_DIM
	print('\nprediction\nrow: %d, col: %d, height: %d, width: %d' % (int(p[0]), int(p[1]), int(p[2]), int(p[3])))

	plot_prediction(x, p)



def plot_prediction(x, p):
	# draw the box:
	fig, ax = plt.subplots(1)
	ax.imshow(x)

	# need to specify [col, row, width, height]
	rect = Rectangle(
		(int(p[1]), int(p[0])),
		int(p[3]), int(p[2]), 
		linewidth=1, edgecolor='r', facecolor='none'
	)

	ax.add_patch(rect)
	plt.show()



def main():
	# load the object image:
	ob = image.load_img('pikachu_tight.png')
	# compare the two ways of loading:
	# ob_ = imread('pikachu_tight.png') # (ob_H x ob_W x 4) b/c includes includes the alpha channel
	# plt.figure(10)
	# plt.imshow(ob)
	# plt.title(str(type(ob))+'\n'+str(np.array(ob).shape))
	# plt.figure(11)
	# plt.imshow(ob_)
	# plt.title(str(type(ob_))+'\n'+str(ob_.shape))
	# # plot the alpha channel:
	# plt.figure(12)
	# plt.imshow(ob_[:,:,-1], cmap='gray')
	# plt.title('object image transparency')
	# # plot the histogram:
	# plt.figure(13)
	# plt.hist(ob_[:,:,-1].flatten())
	# plt.title('histogram of the alpha channel')
	# # print the unique values of the alpha channel:
	# print(set(ob_[:,:,-1].flatten()))
	# plt.show()
	# exit()

	# create the model:
	model = make_model(loss='mse')

	# sanity check - test the generator:
	gen = image_generator(np.array(ob), 1)
	# X, Y = next(gen)
	# x, y = X[0], Y[0]
	# plt.figure(0)
	# plt.imshow(x)
	# plt.title('row: {}, col: {}, height: {}, width: {}'.format(IMG_DIM*y[0], IMG_DIM*y[1], IMG_DIM*y[2], IMG_DIM*y[3]))
	# plt.show()
	# exit()

	# gen_2 = image_generator(np.array(ob), 1)
	# X, Y = next(gen_2)
	# x, y = X[0], Y[0]
	# plt.figure(1)
	# plt.imshow(x)
	# plt.title('row: {}, col: {}, height: {}, width: {}'.format(IMG_DIM*y[0], IMG_DIM*y[1], IMG_DIM*y[2], IMG_DIM*y[3]))
	# plt.show()
	# exit()


	# pass the data generator to our model and train the model:
	model.fit_generator(
		image_generator(np.array(ob), 16, 50), 
		steps_per_epoch=50,
		epochs=5,
	)

	for _ in range(10):
		X, Y = next(gen)
		make_and_plot_prediction(model, X[0], Y[0])




if __name__ == '__main__':
	main()