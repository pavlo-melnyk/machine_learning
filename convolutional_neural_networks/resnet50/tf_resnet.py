import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import keras 

from keras.applications.resnet50 import ResNet50 
from keras.models import Model 
from keras.preprocessing import image
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input, decode_predictions

from tf_resnet_convblock import ConvLayer, BatchNormLayer, ConvBlock 
from tf_resnet_identity_block import IdentityBlock
from tf_resnet_first_layers import ReLULayer, MaxPoolLayer

# Recall the architecture of ResNet50:
#  ConvBlock
#  IdentityBlock x 2
#
#  ConvBlock
#  IdentityBlock x 3
#
#  ConvBlock
#  IdentityBlock x 5
#
#  ConvBlock
#  IdentityBlock x 2
#
#  AveragePooling2D
#  Flatten
#  Dense (Softmax)


# define the additional layers, so they will 
# have a forward function as well as the others:

class AvgPool:
	def __init__(self, kernel_size):
		self.kernel_size = kernel_size

	def forward(self, X):
		return tf.nn.avg_pool(
			X, 
			ksize=[1, self.kernel_size, self.kernel_size, 1],
			strides=[1, 1, 1, 1],
			padding='VALID'
		)

	def get_params(self):
		return []



class Flatten:
	def forward(self, X):
		return tf.contrib.layers.flatten(X)

	def get_params(self):
		return []



def custom_softmax(a):
	'''Customized softmax: exponentiating big values
	may result in numerical overflow.'''

	# a is of NxK size
	a_max = tf.reduce_max(a, axis=1) # result is an Nx1 matrix
	a = a - a_max # discard the max value of each string from every element of that string
	exp_a = tf.exp(a)
	return exp_a / tf.reduce_sum(exp_a)



class DenseLayer:
	def __init__(self, m_in, m_out):
		self.W = tf.Variable((np.random.randn(m_in, m_out) / np.sqrt(2.0 / m_in)).astype(np.float32))
		self.b = tf.Variable(np.zeros(m_out, dtype=np.float32))

	def forward(self, X):
		return tf.matmul(X, self.W) + self.b

	def copyFromKerasLayers(self, layer):
		W, b = layer.get_weights()
		op1 = self.W.assign(W)
		op2 = self.b.assign(b)
		self.session.run((op1, op2))

	def get_params(self):
		return [self.W, self.b]



class TFResNet:
	def __init__(self):
		self.layers = [
			# before the first convblock:
			ConvLayer((7, 7), c_in=3, c_out=64, stride=2, padding='SAME'),
			BatchNormLayer(64),
			ReLULayer(),
			MaxPoolLayer(dim=3),

			# convblock:
			ConvBlock(c_in=64, fm_sizes=[64, 64, 256], stride=1),

			# identity block x 2:
			IdentityBlock(c_in=256, fm_sizes=[64, 64, 256]),
			IdentityBlock(c_in=256, fm_sizes=[64, 64, 256]),
			
			# convblock:
			ConvBlock(c_in=256, fm_sizes=[128, 128, 512], stride=2),

			# identity block x 3:
			IdentityBlock(c_in=512, fm_sizes=[128, 128, 512]),
			IdentityBlock(c_in=512, fm_sizes=[128, 128, 512]),
			IdentityBlock(c_in=512, fm_sizes=[128, 128, 512]),

			# convblock:
			ConvBlock(c_in=512, fm_sizes=[256, 256, 1024], stride=2),

			# identity block x 5:
			IdentityBlock(c_in=1024, fm_sizes=[256, 256, 1024]),
			IdentityBlock(c_in=1024, fm_sizes=[256, 256, 1024]),
			IdentityBlock(c_in=1024, fm_sizes=[256, 256, 1024]),
			IdentityBlock(c_in=1024, fm_sizes=[256, 256, 1024]),
			IdentityBlock(c_in=1024, fm_sizes=[256, 256, 1024]),

			# convblock:
			ConvBlock(c_in=1024, fm_sizes=[512, 512, 2048], stride=2),

			# identity block x 2:
			IdentityBlock(c_in=2048, fm_sizes=[512, 512, 2048]),
			IdentityBlock(c_in=2048, fm_sizes=[512, 512, 2048]),

			# pool / flatten / dense:
			AvgPool(kernel_size=7),
			Flatten(),
			DenseLayer(m_in=2048, m_out=1000)
		]

		self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
		self.output = self.forward(self.input_)


	def copyFromKerasLayers(self, layers):
		# conv
		self.layers[0].copyFromKerasLayers(layers[1])
		# bn
		self.layers[1].copyFromKerasLayers(layers[2])
		# cb
		self.layers[4].copyFromKerasLayers(layers[5:17]) # size=12
		# ib x 2
		self.layers[5].copyFromKerasLayers(layers[17:27]) # size=10
		self.layers[6].copyFromKerasLayers(layers[27:37])
		# cb
		self.layers[7].copyFromKerasLayers(layers[37:49])
		# ib x 3
		self.layers[8].copyFromKerasLayers(layers[49:59])
		self.layers[9].copyFromKerasLayers(layers[59:69])
		self.layers[10].copyFromKerasLayers(layers[69:79])
		# cb
		self.layers[11].copyFromKerasLayers(layers[79:91])
		# ib x 5
		self.layers[12].copyFromKerasLayers(layers[91:101])
		self.layers[13].copyFromKerasLayers(layers[101:111])
		self.layers[14].copyFromKerasLayers(layers[111:121])
		self.layers[15].copyFromKerasLayers(layers[121:131])
		self.layers[16].copyFromKerasLayers(layers[131:141])
		# cb
		self.layers[17].copyFromKerasLayers(layers[141:153])
		# ib x 2
		self.layers[18].copyFromKerasLayers(layers[153:163])
		self.layers[19].copyFromKerasLayers(layers[163:173])
		# dense
		self.layers[22].copyFromKerasLayers(layers[175])


	def forward(self, X):
		for layer in self.layers:
			X = layer.forward(X)
		return X


	def predict(self, X):
		assert(self.session is not None)
		return self.session.run(
			self.output,
			feed_dict={self.input_: X}
		)


	def set_session(self, session):
		self.session = session
		for layer in self.layers:
			if isinstance(layer, ConvBlock) or isinstance(layer, IdentityBlock):
				layer.set_session(session)
			else:
				layer.session = session


	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()


if __name__ == '__main__':
	resnet_ = ResNet50(weights='imagenet')

	# let's make a new resnet without the softmax:
	x = resnet_.layers[-2].output
	W, b = resnet_.layers[-1].get_weights()
	y = Dense(1000)(x)
	resnet = Model(resnet_.input, y)
	resnet.layers[-1].set_weights([W, b])

	partial_model = Model(
		inputs=resnet.input,
		outputs=resnet.layers[175].output
	)

	# always usel when building a model:
	print(partial_model.summary())

	# now let's make an instance of our own model:
	# (again, 'partial' is because we chop off the softmax output)
	my_partial_resnet = TFResNet()

	# let's make a fake image:
	X = np.random.random((1, 224, 224, 3)).astype(np.float32)

	# get keras output:
	keras_output = partial_model.predict(X)

	# get our model output:
	# 1) let's intialize only the varialbles in our network:
	init = tf.variables_initializer(my_partial_resnet.get_params())

	# note: starting a new session will mess up the Keras model:
	session = keras.backend.get_session()
	my_partial_resnet.set_session(session)
	session.run(init)

	# first, just make sure we can get any output:
	first_output = my_partial_resnet.predict(X)
	print('first_output.shape:', first_output.shape)

	# now let's copy params from the Keras model:
	my_partial_resnet.copyFromKerasLayers(partial_model.layers)

	# compare the 2 models:
	output = my_partial_resnet.predict(X)
	diff = np.abs(output - keras_output).sum()
	if diff < 1e-10:
		print('Everything\'s great!')
	else:
		print('diff = %s' % diff)