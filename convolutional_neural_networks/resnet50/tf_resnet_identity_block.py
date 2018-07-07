import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

from tf_resnet_convblock import ConvLayer, BatchNormLayer


class IdentityBlock:
	def __init__(self, c_in, fm_sizes, activation=tf.nn.relu):
		# and IdentityBlock consists of 3 ConvLayers:
		# conv1, conv2, conv3
		assert(len(fm_sizes) == 3)

		self.session = None
		self.activation = activation

		# note: stride is always 1

		# init main branch:
		# Conv -> BN -> activation ---> Conv -> BN -> activation ---> Conv -> BN
		self.conv1 = ConvLayer((1, 1), c_in, fm_sizes[0], stride=1)
		self.bn1 = BatchNormLayer(fm_sizes[0])
		self.conv2 = ConvLayer((3, 3), fm_sizes[0], fm_sizes[1], stride=1, padding='SAME')
		self.bn2 = BatchNormLayer(fm_sizes[1])
		self.conv3 = ConvLayer((1, 1), fm_sizes[1], fm_sizes[2], stride=1)
		self.bn3 = BatchNormLayer(fm_sizes[2])

		# for later use:
		self.layers = [
			self.conv1, self.bn1,
			self.conv2, self.bn2,
			self.conv3, self.bn3,
		]

		# next lines won't be used when using whole ResNet:
		self.input_ = tf.placeholder(tf.float32, shape=(1, 224, 224, c_in))
		self.output = self.forward(self.input_)


	def forward(self, X):
		# main branch:
		fX = self.conv1.forward(X)
		fX = self.bn1.forward(fX)
		fX = self.activation(fX)
		fX = self.conv2.forward(fX)
		fX = self.bn2.forward(fX)
		fX = self.activation(fX)
		fX = self.conv3.forward(fX)
		fX = self.bn3.forward(fX)
		
		# shortcut is just input data X
		return self.activation(fX + X)


	def predict(self, X):
		# we need to run the prediction in a session:
		assert(self.session is not None)
		return self.session.run(
			self.output,
			feed_dict={self.input_: X}
		)


	def set_session(self, session):
		# we need to set a session on every layer/sub-layer:
		self.session = session
		self.conv1.session = session
		self.bn1.session = session 
		self.conv2.session = session 
		self.bn2.session = session 
		self.conv3.session = session 
		self.bn3.session = session 


	def copyFromKerasLayers(self, layers):
		assert(len(layers) == 10)
		# <keras.layers.convolutional.Conv2D at 0x7fa44255ff28>,
		# <keras.layers.normalization.BatchNormalization at 0x7fa44250e7b8>,
		# <keras.layers.core.Activation at 0x7fa44252d9e8>,
		# <keras.layers.convolutional.Conv2D at 0x7fa44253af60>,
		# <keras.layers.normalization.BatchNormalization at 0x7fa4424e4f60>,
		# <keras.layers.core.Activation at 0x7fa442494828>,
		# <keras.layers.convolutional.Conv2D at 0x7fa4424a2da0>,
		# <keras.layers.normalization.BatchNormalization at 0x7fa44244eda0>,
		# <keras.layers.merge.Add at 0x7fa44245d5c0>,
		# <keras.layers.core.Activation at 0x7fa44240aba8>
		self.conv1.copyFromKerasLayers(layers[0])
		self.bn1.copyFromKerasLayers(layers[1])
		self.conv2.copyFromKerasLayers(layers[3])
		self.bn2.copyFromKerasLayers(layers[4])
		self.conv3.copyFromKerasLayers(layers[6])
		self.bn3.copyFromKerasLayers(layers[7])


	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params



if __name__ == '__main__':
	identity_block = IdentityBlock(c_in=256, fm_sizes=[64, 64, 256])

	# make a fake image
	X = np.random.random((1, 224, 224, 256))

	init = tf.global_variables_initializer()
	with tf.Session() as session:
		identity_block.set_session(session)
		session.run(init)

		output = identity_block.predict(X)
		print("output.shape:", output.shape)