import gym 
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime


class FeatureTransformer:
	def __init__(self, n_bins):
			self.n_bins = n_bins
			self.cart_position_bins = np.linspace(-2.4, 2.4, n_bins-1)
			self.cart_velocity_bins = np.linspace(-2, 2, n_bins-1)
			self.pole_angle_bins = np.linspace(-0.4, 0.4, n_bins-1)
			self.pole_velocity_bins = np.linspace(-3.5, 3.5, n_bins-1)

	def transform(self, state):
			# assigns a unique int to a state
			cart_pos, cart_vel, pole_angle, pole_vel = state 			
			quantized = [np.digitize([cart_pos], self.cart_position_bins)[0],
						 np.digitize([cart_vel], self.cart_velocity_bins)[0],
						 np.digitize([pole_angle], self.pole_angle_bins)[0],
						 np.digitize([pole_vel], self.pole_velocity_bins)[0],
						]
			return int(''.join(str(int(v)) for v in quantized))



class Model:
	def __init__(self, env, feature_transformer, lr):
		self.env = env 
		self.feature_transformer = feature_transformer
		self.lr = lr

		l = len(str(feature_transformer.n_bins-1)) # number of figures in n_bins	
		n_states = 10**(l*env.observation_space.shape[0])  
		n_actions = env.action_space.n

		# the action-value function is stored as an array:
		self.Q = np.zeros((n_states, n_actions))
		# self.Q = np.random.uniform(low=-1, high=1, size=(n_states, n_actions))

	def predict(self, s):
		idx = self.feature_transformer.transform(s)
		# print('self.Q[idx].shape:', self.Q[idx].shape)
		# exit()
		return self.Q[idx] # (n_actions,)

	def update(self, s, a, G):
		idx = self.feature_transformer.transform(s)
		self.Q[idx, a] += self.lr*(G - self.Q[idx, a])

	def eps_greedy_action(self, s, eps):
		# epsilon-soft
		p = np.random.random()
		if p >= eps:
		      q_s = self.predict(s) # (n_actions, )
		      return np.argmax(q_s)
		else:
		      return self.env.action_space.sample()
		


def play_game(model, eps, gamma, display=False):
	s = model.env.reset() # the start state
	done = False
	total_r = 0
	i = 0 # number of steps per episode
	states = []
	while not done:
		if display:
			# watch the episode:
			model.env.render()
		states.append(s)	
		# do epsilon-greedy:
		a = model.eps_greedy_action(s, eps)

		# take the action, land in a new state, receive a reward:   
		s_prime, r, done, _ = env.step(a)

		total_r += r
		
		# select the best q_prime:      
		max_q_prime = np.max(model.predict(s_prime))
		
		# if done and i < 199: # for 'CartPole-v0'
		if done and i < 499:
			# penalize the Agent for not reaching
			# the max episode duration:
			r = -300

		# Q-learning update:
		G = r + gamma*max_q_prime # the estimate of the return
		model.update(s, a, G)

		i += 1
						
		# the next state becomes current:
		s = s_prime	

	# ass the terminal state:
	states.append(s)	
	return i, total_r, states



def plot_running_avg(total_rewards):
	N = len(total_rewards)
	running_avg = np.empty(N)
	for t in range(N):
		# average over 100 episodes
		# (as per Open AI docs, the Agent is judged by how well
		# it has performed over 100 episodes):
		running_avg[t] = total_rewards[max(0, t-100):t+int(t==0)].mean()
	plt.plot(running_avg)
	plt.title('Total Rewards Running Average Over 100 Episodes')
	plt.show()



if __name__ == '__main__':
	# get the environment:
	# env = gym.make('CartPole-v0')
	env = gym.make('CartPole-v1') # https://gym.openai.com/envs/CartPole-v1
	
	filename = os.path.basename(__file__).split()[0]
	save_as = filename + ' ' + str(datetime.now()).replace(':', ' ') # enable for windows

	if 'wrap' in sys.argv:
		env = wrappers.Monitor(env, save_as)

	# define our model:
	lr = 1e-1 # the learning rate
	state2idx = FeatureTransformer(n_bins=10)	
	model = Model(env, state2idx, lr)
					
	# play the game:
	n_episodes = 20000
	gamma = 0.9 # the discount factor

	avg_length = 0
	total_rewards = np.empty(n_episodes)
	states = []
	t0 = datetime.now()	

	for t in range(n_episodes):
		# decay epsilon:
		eps = 1.0/np.sqrt(t+1) 
		
		# play an episode:
		i, total_r, visited_states = play_game(model, eps, gamma)

		# update the average:
		avg_length += (i - avg_length)/(t+1)

		total_rewards[t] = total_r
		states += visited_states
		
		if t % 100 == 0:
			print('t:', t, '\ttotal_r:', total_r, '\tavg_r last 100:',\
				  np.round(total_rewards[max(0, t-100):t+int(t==0)].mean(), 4), '\teps:', np.round(eps, 5))

	dt = datetime.now() - t0

	env.close()

	print('\nETA:', dt)
	print('\navg episode length:', avg_length)	
	print('\navg rewards for last 100 episodes:', total_rewards[-100:].mean())
	print('\nnumber of quantized states visited:', np.array(model.Q!=0).sum()) # number of non-zero entries of Q
	print('\nnumber of continuous states visited:', len(states))

	plt.plot(total_rewards)
	plt.title('Total Rewards')
	plt.xlabel('episodes')
	plt.show()

	plot_running_avg(total_rewards)	

	
	
	