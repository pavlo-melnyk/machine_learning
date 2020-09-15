# Object Localization project from  
# https://www.udemy.com/course/advanced-computer-vision/
# Step 6: localize an object of different orientation and sizes on different b/g
#         assuming that the object may not appear in an image at all.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

if tf.__version__[0] == '2':
	# for TF 2.0 and newer:
	print('Hello TensorFlow {}!'.format(tf.__version__))
	from tensorflow.keras.applications import VGG16
	from tensorflow.keras.applications.vgg16 import preprocess_input
	from tensorflow.keras.layers import Flatten, Dense
	from tensorflow.keras.models import Model 
	from tensorflow.keras.optimizers import Adam, SGD
	from tensorflow.keras.losses import binary_crossentropy
	from tensorflow.keras.preprocessing import image

else:
	import keras
	from keras.applications import VGG16
	from keras.applications.vgg16 import preprocess_input
	from keras.layers import Flatten, Dense
	from keras.models import Model 
	from keras.optimizers import Adam, SGD
	from keras.losses import binary_crossentropy
	from keras.preprocessing import image

from imageio import imread
from skimage.transform import resize
from matplotlib.patches import Rectangle 
from glob import glob



IMG_DIM = 200

# weights for the custom loss components:
ALPHA = 2
BETA = 0.5



def custom_loss(y_true, y_pred):
	# y_true[i] = (row, col, height, width, p(object_appeared|img))
	# the bounding box loss:
	bce_1 = binary_crossentropy(y_true[:, :-1], y_pred[:, :-1])
	# the binary prediction (about an object being present in an image) loss:
	bce_2 = binary_crossentropy(y_true[:, -1], y_pred[:, -1])
	return ALPHA * y_true[:, -1] * bce_1 + BETA * bce_2



def make_model(loss=custom_loss, lr=1e-4):
	vgg = VGG16(
		input_shape=[IMG_DIM, IMG_DIM, 3],
		include_top=False,
		weights='imagenet')
	x = Flatten()(vgg.output)
	x = Dense(5, activation='sigmoid')(x)
	model = Model(vgg.input, x)
	model.compile(loss=loss, optimizer=Adam(lr=lr))	
	return model



def image_generator(ob_img, bg_imgs, batch_size=64, n_batches=10):
	# generate IMG_DIMxIMG_DIM samples and targets:

	# IMG_DIM = bg_image.shape[:2]
	
	ob_H, ob_W = ob_img.shape[:2]

	while True:
		for _ in range(n_batches):
			# create placeholders:
			# X = np.repeat(np.expand_dims(bg_image, 0), batch_size, 0)
			X = np.zeros((batch_size, IMG_DIM, IMG_DIM, 3))
			Y = np.zeros((batch_size, 5))

			for i in range(batch_size):
				# select a b/g image:
				bg_img = bg_imgs[np.random.choice(len(bg_imgs))]
				
				# reshape the b/g image:
				# bg_img_new = resize(
				# 	bg_img, 
				# 	(IMG_DIM, IMG_DIM), 
				# 	preserve_range=True).astype(np.uint8)

				# alternatively, crop an IMG_DIMxIMG_DIM region from the b/g image:
				bg_H, bg_W, _ = bg_img.shape
				assert bg_H >= IMG_DIM and bg_W >= IMG_DIM, 'b/g image must be at least (%d,%d)' % (IMG_DIM, IMG_DIM)
				try:
					r_H = np.random.randint(bg_H - IMG_DIM)      
					r_W = np.random.randint(bg_W - IMG_DIM)
				except ValueError:
					# in case bg_H == IMG_DIM
					r_H = 0
					r_W = 0

				# place the b/g:
				X[i, :, :, :] = bg_img[r_H:r_H+IMG_DIM, r_W:r_W+IMG_DIM, :3].copy()

				# object may not appear in an image:
				p_appear = np.random.random()

				if p_appear > 0.5:					
				# resize the object:
					scale = np.random.uniform(0.5, 1.5)
					# scale = 0.5 + np.random.random() # [0.5, 1.5]
					ob_H_new, ob_W_new = int(scale * ob_H), int(scale * ob_W)
					ob_img_new = resize(
						ob_img,
						(ob_H_new, ob_W_new),
						preserve_range=True).astype(np.uint8) # 0...255

					# select a location for the object:
					row0 = np.random.randint(IMG_DIM - ob_H_new)
					col0 = np.random.randint(IMG_DIM - ob_W_new)
					row1 = row0 + ob_H_new # row1 >= row0
					col1 = col0 + ob_W_new # col1 >= col0
					
					# with probability 0.5, flip the object:
					if np.random.random() < 0.5:
						ob_img_new = np.fliplr(ob_img_new)

					# extract the transparency information from the object image：
					mask = np.expand_dims(ob_img_new[:,:,-1], -1) == 0

					# "crop" the space for the object:
					X[i, row0:row1, col0:col1, :] *= mask

					# place the object:				
					X[i, row0:row1, col0:col1, :] += ob_img_new[:,:,:3]

					
					# normalize the targets to be in range [0, 1]:
					Y[i, 0] = row0 / IMG_DIM            # top-left corner y-coord
					Y[i, 1] = col0 / IMG_DIM            # tor-left corner x-coord
					Y[i, 2] = (row1 - row0) / IMG_DIM   # height
					Y[i, 3] = (col1 - col0) / IMG_DIM   # width
				
				# the binary decision p(object_appeared|img) = {0, 1}:
				Y[i, 4] = p_appear > 0.5
								
			# yield a batch of samples and targets:
			yield X / 255., Y



def make_and_plot_prediction(model, x, y=''):
	if len(y) == 5:
		y *= IMG_DIM
		print('\n\ntarget\nrow: %d, col: %d, height: %d, width: %d, p(object_appeared|img): %d' % (int(y[0]), int(y[1]), int(y[2]), int(y[3]), int(y[4])))

	# predict bounding box using the pre-trained model:
	p = model.predict(np.expand_dims(x, axis=0))[0]

	# reverse the transformation into un-normalized form:
	p *= IMG_DIM
	print('\nprediction\nrow: %d, col: %d, height: %d, width: %d, p(object_appeared|img): %d' % (int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])))

	plot_prediction(x, p)



def plot_prediction(x, p):
	# draw the box:
	fig, ax = plt.subplots(1)
	ax.imshow(x)
	# if the object is detected:
	if p[4] > 0.5:
		# need to specify [col, row, width, height]
		rect = Rectangle(
			(p[1], p[0]),
			p[3], p[2], 
			linewidth=1, edgecolor='r', facecolor='none'
		)
		ax.add_patch(rect)
	else:
		plt.title('No object')
	plt.show()

# from ol_step_2 import image_generator



def main():
	# load the object image:
	ob = imread('bulbasaur_tight.png')
	# ob = imread('pikachu_tight.png')
	# ob = imread('charmander_tight.png')
	ob = np.array(ob)
	
	# plt.figure(10)
	# plt.imshow(ob)
	# plt.title(str(type(ob))+'\n'+str(ob.shape))
	# plt.show()
	# exit()
	
	# load b/g images:
	bg_imgs = []
	bg_files = glob('backgrounds/*.jpg')
	for bg in bg_files:
		bg_imgs.append(np.array(imread(bg)))

	# create the model:
	model = make_model(loss=custom_loss, lr=1e-4)

	# sanity check - test the generator:
	gen = image_generator(ob, bg_imgs, 1)
	for _ in range(10):
		X, Y = next(gen)
		x, y = X[0], (IMG_DIM * Y[0]).astype(np.int32)	
		plot_prediction(x, y)	
	# exit()

	# pass the data generator to our model and train the model:
	model.fit_generator(
		image_generator(ob, bg_imgs, 16, 50), 
		steps_per_epoch=50,
		epochs=5,
	)

	for _ in range(10):
		X, Y = next(gen)
		make_and_plot_prediction(model, X[0], Y[0])



if __name__ == '__main__':
	main()