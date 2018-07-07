import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import keras

from keras.applications.resnet50 import ResNet50 


np.random.seed(1996)

def init_filter(shape):
	'''tf filter shape be (width, height, num_chnl_in, num_chnl_out)'''
	return np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1])/2.0).astype(np.float32)



class ConvLayer:
	def __init__(self, f_size, c_in, c_out, stride=2, padding='VALID'):
		self.W = tf.Variable(init_filter((*f_size, c_in, c_out)).astype(np.float32))
		bias_init = np.zeros(c_out, dtype=np.float32)
		self.b = tf.Variable(bias_init)
		self.stride = stride
		self.padding = padding

	def forward(self, X):
		X = tf.nn.conv2d(
			X,
			self.W,
			strides=[1, self.stride, self.stride, 1],
			padding=self.padding
		)
		return X + self.b

	def copyFromKerasLayers(self, layer):
		'''Used later to pass in real weight from Keras's pretrained ResNet-50'''
		# only 1 layer to copy from:
		W, b = layer.get_weights()
		op1 = self.W.assign(W)
		op2 = self.b.assign(b)
		self.session.run((op1, op2))

	def get_params(self):
		return [self.W, self.b]



class BatchNormLayer:
	def __init__(self, D):
		'''
		We assume two modes - 'training' and 'test'.

		For the 'training mode' we calculate the sample mean and the sample std of a current batch.
		We then normalize the data by subtracting from it the sample mean and dividing by the std.
		Afterwards, we should scale the data to something else, giving it new mean and different std
		(gamma and beta - second scale and location parameters, respectively).

		For the 'test' mode we need to keep track of all mean and variance (or std) values during the
		training step. And then we use them for standardizing test samples.
		'''
		self.gamma = tf.Variable(np.ones(D, dtype=np.float32))
		self.beta = tf.Variable(np.zeros(D, dtype=np.float32))
		self.rn_mean = tf.Variable(np.zeros(D, dtype=np.float32), trainable=False)
		self.rn_var = tf.Variable(np.ones(D, dtype=np.float32), trainable=False)

	def forward(self, X, is_training=False, decay=0.9):
		if is_training:
			# get the current batch mean and variance:
			batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2])
			# update the running mean and running variance:
			update_rn_mean = tf.assign(
				self.rn_mean,
				self.rn_mean*decay + (1 - decay)*batch_mean
			)

			update_rn_var = tf.assign(
				self.rn_var, 
				self.rn_var*decay + (1 - decay)*batch_var
			)

			# to make sure the aforementioned updates are calculated
			# every time we call the train function,
			# we have to use next function:
			with tf.control_dependencies([update_rn_mean, update_rn_var]):
				X = tf.nn.batch_normalization(
					X,
					batch_mean, 
					batch_var, 
					self.beta, 
					self.gamma, 
					1e-3
			)

		else:
			# test mode:
			X = tf.nn.batch_normalization(
				X, 
				self.rn_mean,
				self.rn_var, 
				self.beta, 
				self.gamma,
				1e-3
			)

		return X

	def copyFromKerasLayers(self, layer):
		# only 1 layer to copy from
		# order:
		# gamma, beta, moving mean, moving variance
		gamma, beta, rn_mean, rn_var = layer.get_weights()
		op1 = self.rn_mean.assign(rn_mean)
		op2 = self.rn_var.assign(rn_var)
		op3 = self.gamma.assign(gamma)
		op4 = self.beta.assign(beta)
		self.session.run((op1, op2, op3, op4))

	def get_params(self):
		return [self.rn_mean, self.rn_var, self.gamma, self.beta]



class ConvBlock:
	def __init__(self, c_in, fm_sizes, stride=2, activation=tf.nn.relu):
		'''Build a ConvBlock of ResNet-50. 
		All sizes are from original paper: https://arxiv.org/abs/1512.03385
		'''
		# conv1, conv2, conv3
		# fm_sizes - feature map sizes - are basically numbers of filters for each Conv output
		assert (len(fm_sizes)==3)

		# note: kernel size in 2nd conv is always 3

		# note: stride only applies to conv1 in main branch
		#      and conv in shortcut, otherwise stride is 1
		self.session = None
		self.activation = activation

		# init main branch:
		# Conv -> BN -> activation ---> Conv -> BN -> activation ---> Conv -> BN
		self.conv1 = ConvLayer((1, 1), c_in, fm_sizes[0], stride=stride)
		self.bn1 = BatchNormLayer(fm_sizes[0])		
		self.conv2 = ConvLayer((3, 3), fm_sizes[0], fm_sizes[1], stride=1, padding='SAME')
		self.bn2 = BatchNormLayer(fm_sizes[1])
		self.conv3 = ConvLayer((1, 1), fm_sizes[1], fm_sizes[2], stride=1)
		self.bn3 = BatchNormLayer(fm_sizes[2])

		# init shortcut branch:
		# Conv -> BN 
		# note: #feature maps shortcut = #feature maps conv3
		#       因为 they will be summed before the final activation
		#       in the output of the ConvBlock
		self.conv_sc = ConvLayer((1, 1), c_in, fm_sizes[2], stride=stride)
		self.bn_sc = BatchNormLayer(fm_sizes[2])

		# for later use:
		self.layers = [
			self.conv1, self.bn1,
			self.conv2, self.bn2,
			self.conv3, self.bn3,
			self.conv_sc, self.bn_sc
		]

		# We need to define some graph nodes; particularly a placeholder for the input
		# and the output. We need these in order to pass in a np.array (a picture for testing),
		# and get an actual output. This won't be needed for the whole ResNet-50
		# in the complete implementaton.
		self.input_ = tf.placeholder(tf.float32, shape=(1, 224, 224, c_in))
		self.output = self.forward(self.input_)


	def forward(self, X):
		# main branch:
		fX = self.conv1.forward(X)
		# print(fX.get_shape())
		fX = self.bn1.forward(fX)
		# print(fX.get_shape())
		fX = self.activation(fX)
		# print(fX.get_shape())
		fX = self.conv2.forward(fX)
		# print(fX.get_shape())
		fX = self.bn2.forward(fX)
		# print(fX.get_shape())
		fX = self.activation(fX)
		# print(fX.get_shape())
		fX = self.conv3.forward(fX)
		# print(fX.get_shape())
		fX = self.bn3.forward(fX)
		# print(fX.get_shape())

		# shortcut branch:
		sX = self.conv_sc.forward(X)
		# print(sX.get_shape())
		sX = self.bn_sc.forward(sX)
		# print(sX.get_shape())

		return self.activation(fX + sX)

	def predict(self, X):
		assert(self.session is not None)
		return self.session.run(
			self.output,
			feed_dict={self.input_: X} 
			)


	def set_session(self, session):
		# It's going to relevant later on!!!

		# need to make this a session
		# so assingment happens on sublayers too
		self.session = session
		self.conv1.session = session
		self.bn1.session = session
		self.conv2.session = session
		self.bn2.session = session
		self.conv3.session = session
		self.bn3.session = session
		self.conv_sc.session = session
		self.bn_sc.session = session


	def copyFromKerasLayers(self, layers):
		# [<keras.layers.convolutional.Conv2D at 0x117bd1978>,
		#  <keras.layers.normalization.BatchNormalization at 0x117bf84a8>,
		#  <keras.layers.core.Activation at 0x117c15fd0>,
		#  <keras.layers.convolutional.Conv2D at 0x117c23be0>,
		#  <keras.layers.normalization.BatchNormalization at 0x117c51978>,
		#  <keras.layers.core.Activation at 0x117c93518>,
		#  <keras.layers.convolutional.Conv2D at 0x117cc1518>,
		#  <keras.layers.convolutional.Conv2D at 0x117d21630>,
		#  <keras.layers.normalization.BatchNormalization at 0x117cd2a58>,
		#  <keras.layers.normalization.BatchNormalization at 0x117d44b00>,
		#  <keras.layers.merge.Add at 0x117dae748>,
		#  <keras.layers.core.Activation at 0x117da2eb8>]
		self.conv1.copyFromKerasLayers(layers[0])
		self.bn1.copyFromKerasLayers(layers[1])
		self.conv2.copyFromKerasLayers(layers[3])
		self.bn2.copyFromKerasLayers(layers[4])
		self.conv3.copyFromKerasLayers(layers[6])
		self.bn3.copyFromKerasLayers(layers[8])
		self.conv_sc.copyFromKerasLayers(layers[7])
		self.bn_sc.copyFromKerasLayers(layers[9])


	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params 



if __name__ == '__main__':
	conv_block = ConvBlock(c_in=3, fm_sizes=[64, 64, 256], stride=1)

	# model = ResNet50(include_top=True, weights=None, input_tensor=None)
	# print(model.summary())
	# make a fake image:
	X = np.random.random((1, 224, 224, 3)).astype(np.float32)
	# plot the image:
	# print(np.squeeze(X, axis=0).shape)
	plt.imshow(np.squeeze(X, axis=0))
	plt.title('random image')
	plt.show()


	init = tf.global_variables_initializer()
	with tf.Session() as session:
		conv_block.set_session(session)
		session.run(init)

		output = conv_block.predict(X)
		print('output shape:', output.shape)
