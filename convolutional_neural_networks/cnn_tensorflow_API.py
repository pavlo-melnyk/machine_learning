import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
import math
import joblib 

from sklearn.utils import shuffle

from util import getImageData, error_rate, init_weight_and_bias, init_filter_, swish
from datetime import datetime

# image dimensions are expected to be: N x height x width x color
# filter shapes are expected to be: filter height x filter width x input feature maps x output feature maps

class HiddenLayer(object):
	def __init__(self, M1, M2, an_id):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2
		W0, b0 = init_weight_and_bias(M1, M2)
		self.W = tf.Variable(W0, name='W%s' % self.id)
		self.b = tf.Variable(b0, name='b%s' % self.id)
		self.params = [self.W, self.b]

	def forward(self, X):
		return swish(tf.matmul(X, self.W) + self.b, beta=10, tensorflow=True)


class ConvPoolLayer(object):
	def __init__(self, mi, mo, fh=5, fw=5, strides=[1, 1, 1, 1], padding='SAME', poolsz=(2, 2)):
		# mi = input feature map size;
		# mo = output feature map size;
		sz = (fh, fw, mi, mo)
		W0 = init_filter_(sz, mode='tensorflow')
		self.W = tf.Variable(W0)
		b0 = np.zeros(mo, dtype=np.float32)
		self.b = tf.Variable(b0)
		self.s = strides
		self.p = padding
		self.poolsz = poolsz
		self.params = [self.W, self.b]

	def forward(self, X):
		conv_out = tf.nn.conv2d(X, self.W, strides=self.s, padding=self.p)
		# add a bias vector:
		conv_out = tf.nn.bias_add(conv_out, self.b)
		p1, p2 = self.poolsz
		pool_out = tf.nn.max_pool(
			conv_out,
			ksize=[1, p1, p2, 1],
			strides=[1, p1, p2, 1], 
			padding='SAME'
		)
		return swish(pool_out, beta=10, tensorflow=True)


class CNN(object):
	def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
		self.convpool_layer_sizes = convpool_layer_sizes
		self.hidden_layer_sizes = hidden_layer_sizes
		self.session = tf.InteractiveSession()


	def fit(self, X, Y, lr=1e-3, mu=0.990, reg=1e-3, decay=0.99999, eps=1e-10, batch_sz=30, epochs=7, display_cost=False):
		lr = np.float32(lr)
		mu = np.float32(mu)
		reg = np.float32(reg)
		decay = np.float32(decay)
		eps = np.float32(eps)
		K = len(set(Y))
		self.batch_sz = batch_sz

		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)
		Y = Y.astype(np.int32)

		# create a validation set:
		Xvalid, Yvalid = X[-1000:,], Y[-1000:]
		X, Y = X[:-1000,], Y[:-1000]

		# initialize convpool layers:
		N, height, width, c = X.shape
		mi = c
		outh = height 
		outw = width
		self.convpool_layers = []
		for mo, fw, fh in self.convpool_layer_sizes:
			layer = ConvPoolLayer(mi, mo, fw, fh)
			self.convpool_layers.append(layer)
			outh = outh // 2
			outw = outw // 2
			mi = mo

		# initialize mlp (fully-connected) layers:
		self.hidden_layers = []
		M1 = self.convpool_layer_sizes[-1][0]*outh*outw
		count = 0
		for M2 in self.hidden_layer_sizes:
			layer = HiddenLayer(M1, M2, count)
			self.hidden_layers.append(layer)
			M1 = M2
			count += 1

		# the last (logistic regression) layer :
		W, b = init_weight_and_bias(M1, K)
		self.W = tf.Variable(W, name='W%s'%count)
		self.b = tf.Variable(b, name='b%s'%count)

		# collect the params for later use:
		self.params = []
		for layer in self.convpool_layers:
			self.params += layer.params
		for h_layer in self.hidden_layers:
			self.params += h_layer.params
		self.params += [self.W, self.b]

		# set up tensorflow functions and variables:
		self.tfX = tf.placeholder(tf.float32, shape=(None, height, width, c), name='X')
		self.tfY = tf.placeholder(tf.int32, shape=(None, ), name='Y')
		logits = self.forward(self.tfX) # will be defined later

		cost = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=logits,
				labels=self.tfY
			)
		)
		reg_term = reg*sum([tf.nn.l2_loss(p) for p in self.params])
		cost += reg_term

		# define training and prediction operations:
		train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)
		self.predict_op = tf.argmax(logits, 1)
		
		# the main loop:
		n_batches = N // batch_sz
		val_costs = []
		t0 = datetime.now()
		init = tf.global_variables_initializer()	
		self.session.run(init)
		for i in range(epochs):
			X, Y = shuffle(X, Y)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j+1)*batch_sz,]
				Ybatch = Y[j*batch_sz:(j+1)*batch_sz]

				self.session.run(train_op, feed_dict={self.tfX: Xbatch, self.tfY: Ybatch})
				if j % 20 == 0:
					val_cost = self.session.run(cost, feed_dict={self.tfX: Xvalid, self.tfY: Yvalid})
					val_costs.append(val_cost)

					p = self.predict(Xvalid)
					error = error_rate(Yvalid, p)
					print('\ni: %d,  j: %d,  valid_cost: %.3f,  error: %.3f' % (i, j, val_cost, error))

		dt = datetime.now() - t0
		print('\nElapsed time:', dt)

		if display_cost:
			plt.plot(val_costs)
			plt.title('Cost on Validation Set')
			plt.xlabel('iterations')
			plt.ylabel('cost')
			plt.show()


	def forward(self, X):
		Z = X
		for layer in self.convpool_layers:
			Z = layer.forward(Z)
		Z_shape = Z.get_shape().as_list()
		Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
		for h_layer in self.hidden_layers:
			Z = h_layer.forward(Z)
		return tf.matmul(Z, self.W) + self.b


	def predict(self, X):
		return self.session.run(self.predict_op, feed_dict={self.tfX: X})
		

	def score(self, X, Y):
		prediction = np.zeros(len(X))
		for k in range(len(X) // self.batch_sz):
			X_batch = X[k*self.batch_sz:(k+1)*self.batch_sz,]
			prediction[k*self.batch_sz:(k+1)*self.batch_sz] = self.predict(X_batch)
		assert len(Y) == len(prediction)

		return np.mean(Y == prediction)


	def plot_filters(self):
		# create figure with number of subplots corresponding
		# to the number of filters:
		count = 0
		for layer in self.convpool_layers:
			W = layer.params[0].eval() # we need to get real values of filter
			N_filters = W.shape[3]
			N_channels = W.shape[2]
			n = math.ceil(math.sqrt(N_filters))

			fig, axes = plt.subplots(n, n)
			fig.suptitle('filter_%d'%count, fontsize=16)
			fig.subplots_adjust(hspace=0.3, wspace=0.3)
			
			# plot the weights:
			for i, ax in enumerate(axes.flat):
				for j in range(N_channels):
					if i < N_filters:
						img = W[:, :, j, i]
						ax.imshow(img, cmap='gray')
					
					ax.set_xticks([])
					ax.set_yticks([])
			count += 1
					
			plt.show()





def main():
	X, Y = getImageData()
	# X, Y = shuffle(X, Y)
	# Xtest, Ytest = X[-8000:,].astype(np.float32), Y[-8000:].astype(np.int32)
	# X, Y = X[:-8000,].astype(np.float32), Y[:-8000].astype(np.int32)
	
	# transpose the data in the tensorflow's order:
	X = X.transpose((0, 2, 3, 1))
	print('Training set shape:', X.shape)
	# Xtest = Xtest.transpose((0, 2, 3, 1))
	# print('Test set shape:', Xtest.shape)


	model = CNN(
		convpool_layer_sizes=[(32, 3, 3), (64, 3, 3), (128, 3, 3), (256, 3, 3)],
		hidden_layer_sizes=[500, 500],
		)	
	model.fit(X, Y, display_cost=True)

	# joblib.dump(model, 'mymodel.pkl')
	# model = joblib.load('mymodel.pkl')

	train_acc = model.score(X, Y)
	print('\nTraining set acc: %.3f' % train_acc)

	# test_acc = model.score(Xtest, Ytest)
	# print('Test set acc: %.3f' % test_acc)

	# model.plot_filters()




if __name__ == '__main__':
	main()

