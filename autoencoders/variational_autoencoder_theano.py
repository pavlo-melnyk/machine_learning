import numpy as np
import theano
import theano.tensor as T 
import matplotlib.pyplot as plt 
import json

from theano.tensor.shared_randomstreams import RandomStreams
from utils import init_weights_and_biases, get_mnist_data, get_fashion_mnist_data


def rmsprop(cost, params, lr=1e-3, mu=0.0, decay=0.9, eps=1e-10):
	''' Defines updates for RMSProp in Theano. 
	Returns a list of updates.'''
	grads = T.grad(cost, params)
	updates = []
	for p, g in zip(params, grads):
		# cache
		ones = np.ones_like(p.get_value(), dtype=np.float32)
		c = theano.shared(ones)
		new_c = decay*c + (np.float32(1.0) - decay)*g*g

		# momentum:
		zeros = np.zeros_like(p.get_value(), dtype=np.float32)
		v = theano.shared(zeros)
		new_v = mu*v - lr*g / T.sqrt(new_c + eps)

		# param update:
		new_p = p + new_v

		# append the updates:
		updates.append((c, new_c))
		updates.append((v, new_v))
		updates.append((p, new_p))

	return updates


class HiddenLayer:
	def __init__(self, M1, M2, an_id, f=T.nnet.relu):
		W0, b0 = init_weights_and_biases(M1, M2)
		self.W = theano.shared(W0, name='W%s'%an_id)
		self.b = theano.shared(b0, name='b%s'%an_id)
		self.f = f
		self.params = [self.W, self.b]
	def forward(self, X):
		return self.f(X.dot(self.W) + self.b)


class VariationalAutoencoder:
	def __init__(self, D, hidden_layer_sizes, loss='sigmoid_cross_entropy'):
		# hidden_layer_sizes - the sizes of every layer in the ENCODER
		#                      (up to the final hidden layer Z);
		# the DECODER will have the same layers, 
		# but with reverse shapes 
		self.hidden_layer_sizes = hidden_layer_sizes

		# an input batch of thraining data (batch_size x D):
		self.X = T.matrix(name='X_batch')


		################################# ENCODER ################################
		# layers of the ENCODER:
		self.encoder_layers = []
		M1 = D 
		for i, M2 in enumerate(self.hidden_layer_sizes[:-1]):
			h = HiddenLayer(M1, M2, i)
			self.encoder_layers.append(h)
			M1 = M2

		# the final ENCODER layer size 
		# is also the DECODER input layer size:
		M = self.hidden_layer_sizes[-1]

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
		self.std_dev = T.nnet.softplus(Z[:, M:]) + 1e-6 

		# sample from the q(Z| X):
		# (that's why we need the means and the std_dev's)
		# using reparameterization trick:
		# 1) create a standard Normal distribution (N(0, 1))
		# and draw a sample from it:
		self.rng = RandomStreams()
		standard_sample = self.rng.normal((self.means.shape[0], M)) # (batch_size x M)

		# 3) get a sample z ~ q(Z| X) (actually, batch_size samples Z)
		#    by reparameterizing the standard_sample:
		self.Z = self.means + standard_sample * self.std_dev # (batch_size x M)

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
		for j, M2 in enumerate(reversed(self.hidden_layer_sizes[:-1])):
			# index of the layer = the final ENCODER layer index 'i' + new counter 'j' + 1:
			ind = (j + 1) + i + 1
			h = HiddenLayer(M1, M2, ind) 
			self.decoder_layers.append(h)
			M1 = M2

		# the DECODER final layer normally has a sigmoid activation,
		# thus gives us the final output as probabilities:
		h = HiddenLayer(M1, D, ind + 1, f=T.nnet.sigmoid)
		self.decoder_layers.append(h)

		# get the posterior predictive
		# (perform the forward pass through the DECODER):
		Z = self.Z # here Z stands for a current layer value
		for layer in self.decoder_layers:
			Z = layer.forward(Z)
		self.posterior_predictive_probs = Z # for later use, size = (batch_size x D)

		# recall that output of the DECODER is a distribution;
		# draw the POSTERIOR PREDICTIVE SAMPLE - sample from DECODER's output:
		self.posterior_predictive = self.rng.binomial(
			size=self.posterior_predictive_probs.shape,
			n=1, 
			p=self.posterior_predictive_probs
		) # X_reconstructed (D, )

		# draw the PRIOR PREDICTIVE SAMPLE:
		# 1) draw a sample from the standard normal - Z ~ N(0, 1):
		Z_standard = self.rng.normal((1, M))
		# 2) pass it through the decoder:
		Z = Z_standard
		for layer in self.decoder_layers:
			Z = layer.forward(Z)		
		self.prior_predictive_probs = Z

		# 3) the output is a Bernoulli distribution - p(x_reconstructed| Z_standard);
		#    draw a sample from it:
		self.prior_predictive = self.rng.binomial(
			size=self.prior_predictive_probs.shape,
			n=1, 
			p=self.prior_predictive_probs
		)

		# PRIOR PREDICTIVE GIVEN Z PROBS; used for visualization later on:
		# create a placeholder for the input to the decoder:
		self.Z_input = T.matrix('Z_input')
		Z = self.Z_input # here Z stands for a current layer value
		for layer in self.decoder_layers:
			Z = layer.forward(Z)
		self.prior_predictive_probs_given_Z = Z


		################################# COST ##################################
		# the ELBO - our objective for maximization - consists of two parts:
		# ELBO = expected_log_likelihood - KL-divergence;
		# but we'll minimize -ELBO instead:
		# -ELBO = -expected_log_likelihood - (-kl_divergence)

		# 1) KL-divergence:
		#   we are comparing our q_of_z_given_x - a Gaussian (actually, batch_size Gaussians) -
		# 	with the standard normal
		kl = -T.log(self.std_dev) + 0.5*(self.std_dev**2 + self.means**2) - 0.5 # (batch_size x M)
		kl = T.sum(kl, axis=1) # (batch_size x 1) - we get the kl-divergence per sample

		# 2) expected log-likelihood (negative cross-entropy):
		expected_log_likelihood = -T.nnet.binary_crossentropy(
			output=self.posterior_predictive_probs,
			target=self.X
			)
		expected_log_likelihood = T.sum(expected_log_likelihood, axis=1) # (batch_size x 1)
		
		self.elbo = T.sum(expected_log_likelihood - kl)


		#################### TRAINING AND PREDICTION FUNCTIONS ##################
		# 1) collect the params:
		params = []
		for layer in self.encoder_layers:
			params += layer.params
		for layer in self.decoder_layers:
			params += layer.params

		# 2) define the updates:
		decay = np.float32(0.9)
		learning_rate = np.float32(0.001)

		updates = rmsprop(-self.elbo, params, lr=1e-3, decay=decay)
		

		# training operation: 
		self.train_op = theano.function(
			inputs=[self.X],
			outputs=self.elbo,
			updates=updates
		)
		
		# NOTE: here and later x_reconstructed is called so for analogy with a simple autoencoder,
		# 		it actually means a new drawn sample:
		# draw a sample from p(x_reconstructed | X)
		self.posterior_predictive_sample_and_probs = theano.function(
			inputs=[self.X],
			outputs=[self.posterior_predictive, self.posterior_predictive_probs]
		)

		# draw a sample from p(x_reconstructed | z), where z ~ N(0, 1)
		self.prior_predictive_sample_and_probs = theano.function(
			inputs=[],
			outputs=[self.prior_predictive, self.prior_predictive_probs]
		)

		# return mean of q(z | x)
		# (mean of our VAE approximation of true posterior p(z | x)):
		self.transform = theano.function(
			inputs=[self.X],
			outputs=self.means
		)

		# draw a sample form p(x_reconstructed | z), from a given z
		# (actually returns mean)
		self.prior_predictive_probs_given_input = theano.function(
			inputs=[self.Z_input],
			outputs=self.prior_predictive_probs_given_Z
		)


	def fit(self, X, epochs=30, batch_size=64):
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
				c = self.train_op(X_batch)
				c /= batch_size
				costs.append(c)
				if j % 100 == 0:
					print('j:%d, cost:%.3f'%(j, c))
		
		# plot the cost:
		plt.plot(costs)
		plt.title('Cost')
		plt.xlabel('epochs')
		plt.show()


	def transform(self, X):
		'''Returns output of the encoder.'''
		return self.sess.run(
			self.means,
			feed_dict={self.X: X}
		)


def main():
	X, Y = get_mnist_data(normalize=True)
	# binarize the pictures to get proper Bernoulli random variables
	# (isn't neccessary; one could try both):
	X = (X > 0.5).astype(np.float32)
	Xtest, Ytest = X[-100:], Y[-100:]
	X, Y = X[:-100], Y[:-100]

	# X, Y, Xtest, Ytest = get_fashion_mnist_data(normalize=True)
	# binarize the pictures:
	# X, Xtest = (X > 0.5).astype(np.float32), (Xtest > 0.5).astype(np.float32)
	N, D = X.shape

	vae = VariationalAutoencoder(D, [200, 100])
	vae.fit(X)

	# display original images, posterior predictive samples and probabilities:
	while True:
		i = np.random.choice(len(Xtest))
		x = Xtest[i]
		sample, probs = vae.posterior_predictive_sample_and_probs([x])
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
		image, probs = vae.prior_predictive_sample_and_probs()
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