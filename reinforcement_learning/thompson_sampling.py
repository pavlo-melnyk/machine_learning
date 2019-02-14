'''
An implementation of the Thompson sampling algorithm
(Bayesian method) - 
a solution to the explore-exploit dilemma.
The key is sampling!
'''
import numpy as np 
import matplotlib.pyplot as plt 


class BayesianBandit:
  def __init__(self, true_mean):
      self.true_mean = true_mean # true-mean
				
      # parameters for mu - prior is N(0, 1):
      self.m = 0 # initial value for mean m0 = 0
      self.lambda_ = 1 # initial value for precision lambda0 = 1
      self.sum_x = 0 # sum of the rewards received from this bandit
      self.tau = 1 # precision_of_the_data = 1/variance_of_the_data 

  def pull(self):
      return np.random.randn() + self.true_mean

  @property
  def sample(self):
      '''Generates a Gaussian
      with mean m and precision lambda_ (variance 1/lambda_).'''
      # recall: our mu is a random variable,
      #         so it has its own distribution -
      #         Gaussian (it's our posterior), b/c we assume that 
      #         the likelihood of our data is Gaussian,
      #         and we know that a Gaussian likelihood is
      #         conjugate with a Gaussian prior on the mean;
      #         thus, by choosing a Gaussian prior, the resultant 
      #         posterior distribution of mu is also Gaussian.
      return np.random.randn() / np.sqrt(self.lambda_) + self.m

  def update(self, x):
      # lambda:
      # lambda = lambda0 + tau*N =
      #        = 1 + tau*N

      # mean:
      # mu = (lambda0*m0 + tau*sum_x) / lambda =
      #    = (1*0 + tau*sum_x) / lambda = tau*sum_x / lambda

      self.sum_x += x
      self.lambda_ += self.tau
      self.m = self.tau*self.sum_x / self.lambda_



def run_thompson_sampling_experiment(m1, m2, m3, N):
		bandits = [BayesianBandit(m1), BayesianBandit(m2), BayesianBandit(m3)]

		data = np.empty(N)

		for i in range(N):
				# select a bandit with the highest value of sample of their means:
				bandit = bandits[ np.argmax([b.sample for b in bandits]) ]

				# play the selected bandit = pull the arm and get a reward:
				x = bandit.pull()

				# update the bandit's stats - sum of rewards and the params of the mu distribution:
				bandit.update(x)

				# collect data - all the rewards we received:
				data[i] = x

		# calculate the cumulative average after every play:
		# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.cumsum.html#numpy-cumsum
		# e.g. data = [0.5, 0.11, 0.98], then 
		# cum_sum(data) = [0.5, 0.5+0.11, 0.5+0.11+0.98];
		# to get the cum_avg, we simply divide every entry of cum_sum 
		# by the number of the plays done so far, i.e.
		# cum_avg = [0.5/1, (0.5+0.11)/2, (0.5+0.11+0.98)/3]:
		cumulative_average = np.cumsum(data) / (np.arange(N) + 1) # of shape (N, )

		# plot the moving averages:
		plt.plot(cumulative_average)
		plt.plot(np.ones(N)*m1)
		plt.plot(np.ones(N)*m2)
		plt.plot(np.ones(N)*m3)
		plt.xscale('log')
		plt.show()

		# for bandit in bandits:
		#   print(bandit.m)

		return cumulative_average


if __name__ == '__main__':
		# the true means of every bandit:
		m1, m2, m3 = 1.0, 2.0, 3.0
		# number of plays:
		N = 100000

		ex_1 = run_thompson_sampling_experiment(m1, m2, m3, N)

		# log-scale plot:
		plt.plot(ex_1, label='thompson_sampling')
		plt.legend()
		plt.xscale('log')
		plt.ylabel('average reward')
		plt.xlabel('plays')
		plt.title('Cumulative Averages - Log-Scale Plot')
		plt.show()

		# linear plot:
		plt.plot(ex_1, label='thompson_sampling')
		plt.legend()
		plt.ylabel('average reward')
		plt.xlabel('plays')
		plt.title('Cumulative Averages - Linear Plot')
		plt.show()



