import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
import json

from utils import init_weights_and_biases, get_mnist_data, get_fashion_mnist_data
from collections import Counter

# for later use:
Bernoulli = tf.contrib.distributions.Bernoulli
Normal = tf.contrib.distributions.Normal


class HiddenLayer:
	def __init__(self, M1, M2, an_id, f=tf.nn.relu):
		W0, b0 = init_weights_and_biases(M1, M2)
		self.W = tf.Variable(W0, name="W%s"%an_id)
		self.b = tf.Variable(b0, name="b%s"%an_id)
		self.f = f
		self.params = [self.W, self.b]

	def forward(self, X):
		return self.f(tf.matmul(X, self.W) + self.b)


class VariationalAutoencoder:
	def __init__(self, savefile, D=None, hidden_layer_sizes=None, loss='sigmoid_cross_entropy'):
		self.savefile = savefile
		if D and hidden_layer_sizes:
			self.build(D, hidden_layer_sizes, loss=loss)	

	def build(self, D, hidden_layer_sizes, loss='sigmoid_cross_entropy'):
		tf.reset_default_graph()
		
		# hidden_layer_sizes - the sizes of every layer in the ENCODER
		#                      (up to the final hidden layer Z);
		# the DECODER will have the same layers, 
		# but with reverse shapes 
		
		# an input batch of thraining data (batch_size x D):
		self.X = tf.placeholder(tf.float32, shape=(None, D), name='X_batch')

		################################# ENCODER ################################
		# layers of the ENCODER:
		self.encoder_layers = []
		M1 = D 
		for i, M2 in enumerate(hidden_layer_sizes[:-1]):
			h = HiddenLayer(M1, M2, i)
			self.encoder_layers.append(h)
			M1 = M2

		# the final ENCODER layer size 
		# is also the DECODER input layer size:
		M = hidden_layer_sizes[-1]

		# the ENCODER final layer has no activation;
		# recall that we need twice as many units of the
		# specified size M:
		# M means + M variances = 2M
		h = HiddenLayer(M1, 2 * M, i + 1, f=lambda x: x)
		self.encoder_layers.append(h)

		# to get the mean of the final ENCODER level Z,
		# we don't need any activation function;
		# but to get the variances (std_dev), we need to make sure
		# variances (std_dev) > 0, which is attainable by means
		# of the softplus activation function;

		# get the logits
		# (perform a forward pass through the ENCODER):
		Z = self.X # here Z stands for a current layer value
		for layer in self.encoder_layers:
			Z = layer.forward(Z) # (batch_size x 2*M)
		self.means = Z[:, :M] # a half of the output is the means (batch_size x M)
		# apply softplus and add a small value for smoothing to the other half:
		self.std_dev = tf.nn.softplus(Z[:, M:]) + 1e-6 

		# sample from the q(Z| X):
		# (that's why we need the means and the std_dev's)
		# using reparameterization trick:
		# 1) create a standard Normal distribution (N(0, 1)):
		standard_normal = Normal(
				loc = np.zeros(M, dtype=np.float32),
				scale = np.ones(M, dtype=np.float32)
		)

		# 2) get a sample from the standard normal:
		print(tf.shape(self.means)[0])
		standard_sample = standard_normal.sample(tf.shape(self.means)[0]) # (batch_size x M)

		# 3) get a sample z ~ q(Z| X) (actually, batch_size samples Z)
		#    by reparameterizing the standard_sample:
		self.Z = self.means + standard_sample*self.std_dev # (batch_size x M)

		# the same can be done by feeding our means and std_dev
		# as arguments into the constructor of the Normal class, 
		# and then sampling from it:
		# q_of_Z_given_X = Normal(
		#	loc=self.means,
		#  	scale=self.std_dev, 		
		# )
		# self.Z = q_of_Z_given_X.sample()


		################################# DECODER ################################
		self.decoder_layers = []
		M1 = M # the DECODER input layer size
		for j, M2 in enumerate(reversed(hidden_layer_sizes[:-1])):
			# index of the layer = the final ENCODER layer index (i + 1) + new counter 'j' + 1:
			ind = (j + 1) + i + 1
			h = HiddenLayer(M1, M2, ind) 
			self.decoder_layers.append(h)
			M1 = M2

		# the DECODER final layer normally has a sigmoid activation,
		# thus givin us the final output as probabilities;
		# in this implementation Bernoulli class accepts logits,
		# thus we don't need to use any activation at the final layer:
		h = HiddenLayer(M1, D, ind + 1, f=lambda x: x)
		self.decoder_layers.append(h)

		# get the logits
		# (perform the forward pass through the DECODER):
		Z = self.Z # here Z stands for a current layer value
		for layer in self.decoder_layers:
			Z = layer.forward(Z)
		logits = Z
		posterior_predictive_logits = logits # for later use, size = (batch_size x D)

		# recall that output of the DECODER is a distribution;
		# we ASSUME it's a Bernoulli distribution:
		self.X_hat = Bernoulli(logits=logits) # p(X_reconstructed| X)

		# draw the POSTERIOR PREDICTIVE SAMPLE - sample from X_hat:
		self.posterior_predictive_sample = self.X_hat.sample() # X_reconstructed (D, )
		self.posterior_predictive_probs = tf.nn.sigmoid(logits)

		# draw the PRIOR PREDICTIVE SAMPLE:
		# 1) draw a sample from the standard normal - Z ~ N(0, 1):
		#    we have already defined the standard normal, let's just draw a sample again:
		Z_standard = standard_normal.sample(1) # (M, ) - size of the latent vector
		# 2) pass it through the decoder:
		Z = Z_standard
		for layer in self.decoder_layers:
			Z = layer.forward(Z)
		logits = Z
		# the output Bernoulli distribution:
		prior_predictive_dist = Bernoulli(logits=logits) # p(x_reconstructed| Z_standard)
		self.prior_predictive_sample = prior_predictive_dist.sample()
		self.prior_predictive_probs = tf.nn.sigmoid(logits)

		# PRIOR PREDICTIVE GIVEN Z PROBS; used for visualization later on:
		# create a placeholder for the input to the decoder:
		self.Z_input = tf.placeholder(tf.float32, shape=(None, M))
		Z = self.Z_input # here Z stands for a current layer value
		for layer in self.decoder_layers:
			Z = layer.forward(Z)
		logits = Z
		self.prior_predictive_probs_given_Z = tf.nn.sigmoid(logits)


		################################# SAVE ################################
		# NOTE: variables must exist when calling tf.train.Saver()
		self.saver = tf.train.Saver()
		self.params = []
		for layer in self.encoder_layers+self.decoder_layers:
			self.params += layer.params


		################################# COST ################################
		# the ELBO - our objective for maximization - consists of two parts:
		# ELBO = expected_log_likelihood - KL-divergence;
		# but we'll minimize -ELBO instead:
		# -ELBO = -expected_log_likelihood - (-kl_divergence)

		# 1) KL-divergence:
		#    we are comparing our q_of_z_given_x - a Gaussian (actually, batch_size Gaussians) -
		# with the standard normal
		kl = -tf.log(self.std_dev) + 0.5*(self.std_dev**2 + self.means**2) - 0.5 # (batch_size x M)
		kl = tf.reduce_sum(kl, axis=1) # (batch_size x 1)

		# 2) expected log-likelihood (negative cross-entropy):
		# expected_log_likelihood = -tf.nn.sigmoid_cross_entropy_with_logits(
		# 	logits=posterior_predictive_logits,
		# 	labels=self.X)
		# expected_log_likelihood = tf.reduce_sum(expected_log_likelihood, axis=1) # (batch_size x 1)
		
		# alternatively:
		expected_log_likelihood = self.X_hat.log_prob(self.X) # (batch_size x D)
		expected_log_likelihood = tf.reduce_sum(expected_log_likelihood, axis=1) # (batch_size x 1)

		self.elbo = tf.reduce_sum(expected_log_likelihood - kl)

		
		
	def fit(self, D, hidden_layer_sizes, X, loss='cross_entropy', epochs=15, batch_size=64):
		# define the placeholder, variables and cost:
		self.build(D, hidden_layer_sizes, loss=loss)

		# training operation: 
		train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-self.elbo)

		# set up the session
		sess = tf.InteractiveSession()

		# initialize all the variables in the graph: 
		init = tf.global_variables_initializer()
		sess.run(init)

		costs = []
		n_batches = len(X) // batch_size
		print('n_batches:', n_batches)
		# perform mini-batch stochastic GD:
		for i in range(epochs):
			print('epoch', i)
			# shuffle the data:
			np.random.shuffle(X)
			for j in range(n_batches):
				X_batch = X[j*batch_size:(j+1)*batch_size]
				_, c = sess.run((train_op, self.elbo), feed_dict={self.X: X_batch})
				c /= batch_size
				costs.append(c)
				if j % 100 == 0:
					print('j:%d, cost:%.3f'%(j, c))

		# save the model:
		self.saver.save(sess, self.savefile)
		
		# save for later use:
		self.D = D
		self.hidden_layer_sizes = hidden_layer_sizes
		self.loss_function = loss

		# plot the cost:
		plt.plot(costs)
		plt.title('Cost')
		plt.xlabel('epochs')
		plt.show()


	def transform(self, X):
		'''Returns output of the encoder.'''
		with tf.Session() as sess:
			self.saver.restore(sess, self.savefile)

			return sess.run(
				self.means,
				feed_dict={self.X: X}
			)


	def posterior_predictive(self, X):
		''' Returns reconstructed input - a sample from p(X_reconstructed| X).'''
		with tf.Session() as sess:
			self.saver.restore(sess, self.savefile)
			
			return sess.run(
				(self.posterior_predictive_sample,
				self.posterior_predictive_probs),
				feed_dict={self.X: X}
			)


	def prior_predictive_and_probs(self):
		''' First draws a sample from the chosen prior
		(e.g. from the standard normal - Z ~ N(0, 1)), and then decodes it - 
		draws a sample from p(x_reconstructed| Z), 
		or better to say p(X_new| Z).'''
		with tf.Session() as sess:
			self.saver.restore(sess, self.savefile)
		
			return sess.run((self.prior_predictive_sample,	self.prior_predictive_probs))


	def prior_predictive_probs_given_input(self, Z):
		''' Takes in a latent vector Z, not a sample.
		Generates an image (or output Bernoulli means).'''
		with tf.Session() as sess:
			self.saver.restore(sess, self.savefile)
			
			return sess.run(
				self.prior_predictive_probs_given_Z,
				feed_dict={self.Z_input: Z}
			)


	def get_params(self):
		with tf.Session() as sess:
			self.saver.restore(sess, self.savefile)
			init = tf.global_variables_initializer()
			return sess.run(self.params)


	def save(self, filename):
		j = {
			'D': self.D,
			'hidden_layer_sizes': self.hidden_layer_sizes, 
			'model': self.savefile,
			'loss_function': self.loss_function
		}
		with open(filename, 'w') as f:
			json.dump(j, f)


	@staticmethod
	def load(filename):
		with open(filename) as f:
			j = json.load(f)
		return VariationalAutoencoder(j['model'], j['D'], j['hidden_layer_sizes'], j['loss_function'])



def main():
	X, Y = get_mnist_data(normalize=True)
	# binarize the pictures:
	X = (X > 0.5).astype(np.float32)
	Xtest, Ytest = X[-100:], Y[-100:]
	X, Y = X[:-100], Y[:-100]

	# X, Y, Xtest, Ytest = get_fashion_mnist_data(normalize=True)
	# # binarize the pictures:
	# X, Xtest = (X > 0.5).astype(np.float32), (Xtest > 0.5).astype(np.float32)
	N, D = X.shape

	# create a vae:
	vae = VariationalAutoencoder('./tf.model')
	vae.fit(D, [200, 100], X)

	# print('Print some params:')
	# params = vae.get_params()
	# [print(p[0,:5]) for p in params[::2]]

	# display original images, posterior predictive samples and probabilities: 
	while True:
		i = np.random.choice(len(Xtest))
		x = Xtest[i]
		sample, probs = vae.posterior_predictive([x])
		plt.subplot(1, 3, 1)
		plt.imshow(x.reshape(28, 28), cmap='gray')
		plt.title('Original Image')
		plt.subplot(1, 3, 2)
		plt.imshow(sample.reshape(28, 28), cmap='gray')
		plt.title('Posterior Predictive Sample')
		plt.subplot(1, 3, 3)
		plt.imshow(probs.reshape(28, 28), cmap='gray')
		plt.title('Posterior Predictive Probs')
		plt.show()

		prompt = input('Continue generating?\n')
		if prompt and prompt[0] in ['n', 'N']:
			break

	# display prior predictive samples and probabilities: 
	while True:
		i = np.random.choice(len(Xtest))
		x = Xtest[i]
		image, probs = vae.prior_predictive_and_probs()
		plt.subplot(1, 2, 1)
		plt.imshow(image.reshape(28, 28), cmap='gray')
		plt.title('Prior Predictive Sample')
		plt.subplot(1, 2, 2)
		plt.imshow(probs.reshape(28, 28), cmap='gray')
		plt.title('Prior Predictive Probs')
		plt.show()

		prompt = input('Continue generating?\n')
		if prompt and prompt[0] in ['n', 'N']:
			break

	# save the model:
	print('\nsaving the model.....')
	vae.save('my_trained_vae.json')


	# load the saved model and display samples again:
	print('\nloading the model.....')
	vae = VariationalAutoencoder.load('my_trained_vae.json')
	
	# print('Print some params after loading:')
	# params = vae.get_params()
	# [print(p[0,:5]) for p in params[::2]]

	# display original images, posterior predictive samples and probabilities: 
	while True:
		i = np.random.choice(len(Xtest))
		x = Xtest[i]
		sample, probs = vae.posterior_predictive([x])
		plt.subplot(1, 3, 1)
		plt.imshow(x.reshape(28, 28), cmap='gray')
		plt.title('Original Image')
		plt.subplot(1, 3, 2)
		plt.imshow(sample.reshape(28, 28), cmap='gray')
		plt.title('Posterior Predictive Sample')
		plt.subplot(1, 3, 3)
		plt.imshow(probs.reshape(28, 28), cmap='gray')
		plt.title('Posterior Predictive Probs')
		plt.show()

		prompt = input('Continue generating?\n')
		if prompt and prompt[0] in ['n', 'N']:
			break

	# display prior predictive samples and probabilities: 
	while True:
		i = np.random.choice(len(Xtest))
		x = Xtest[i]
		image, probs = vae.prior_predictive_and_probs()
		plt.subplot(1, 2, 1)
		plt.imshow(image.reshape(28, 28), cmap='gray')
		plt.title('Prior Predictive Sample')
		plt.subplot(1, 2, 2)
		plt.imshow(probs.reshape(28, 28), cmap='gray')
		plt.title('Prior Predictive Probs')
		plt.show()

		prompt = input('Continue generating?\n')
		if prompt and prompt[0] in ['n', 'N']:
			break



if __name__ == '__main__':
	main()