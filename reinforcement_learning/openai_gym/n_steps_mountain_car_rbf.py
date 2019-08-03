import os 
import sys
import gym
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 

from gym import wrappers
from mpl_toolkits.mplot3d import Axes3D 
from q_learning_cart_pole_rbf import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from datetime import datetime 



class FeatureTransformer:
	def __init__(self, env):
		# gather 10^4 samples from the state-space:
		state_samples = np.array([env.observation_space.sample() for i in range(10000)])
		# scale the collected data, s.t. mean = 0, var = 1:
		scaler = StandardScaler()
		scaler.fit(state_samples)
		scaled_samples = scaler.transform(state_samples)

		# a single RBF kernel is 
		# 
		#            f(x) = exp( -gamma * ||x-c||^2 ) ,
		# 
		# where c     - is a center/exemplar/component that lives in
		#               the same space as x;
		#       gamma - a shape parameter.

		# NOTE: RBFSampler from SciKit-Learn just APPROXIMATES the RBF
		#       and doesn't really use training data (input to the fit function);
		#       check the following link for details: 
		#       https://www.kaggle.com/sy2002/rbfsampler-actually-is-not-using-any-rbfs

		# RBF kernels with different variances are to cover different state's data space:
		transformer = FeatureUnion([
			('rbf_1', RBFSampler(gamma=0.5, n_components=500)),
			('rbf_2', RBFSampler(gamma=1.0, n_components=500)),
			('rbf_3', RBFSampler(gamma=2.0, n_components=500)),
			('rbf_4', RBFSampler(gamma=5.0, n_components=500)),
		])

		# in order to investigate the dimensionality,
		# transform the scaled samples collected before:
		example_features = transformer.fit_transform(scaled_samples)
		self.dimensionality = example_features.shape[1]
		self.scaler = scaler
		self.transformer = transformer
		
	def transform(self, states):
		''' States must be a 2D array. '''
		scaled_data = self.scaler.transform(states)
		return self.transformer.transform(scaled_data)



class Model:
	def __init__(self, env, feature_transformer, lr='constant', lr0=0.01):
		self.env = env 
		self.feature_transformer = feature_transformer        
		# we'll have a collection of models - 
		# one model for each action!
		self.rbf_models = []
		for i in range(env.action_space.n):
			rbf_model = SGDRegressor(learning_rate=lr, lr0=lr0)
			# initialize the weights of the linear model - fit some data:
			x = feature_transformer.transform( [env.reset()] )
			rbf_model.partial_fit(x, [0])
			self.rbf_models.append(rbf_model)
			
	def update(self, s, a, G):
		''' Performs a Q-Learning semigradient update.'''
		# transform the state into a feature vector:
		x = self.feature_transformer.transform([s]) # 2D-vector
		# make a gradient descent update for the given model:
		self.rbf_models[a].partial_fit(x, [G]) # targets should be passed as 1D
											   # so we ‘wrap’ the scalar G in a list
		
	def predict(self, s):
		# transform the state into a feature vector:
		x = self.feature_transformer.transform([s]) # data input must be 2D
		return np.array([model.predict(x)[0] for model in self.rbf_models])

	def epsilon_greedy_action(self, s, eps):
		p = np.random.random()
		if p < eps:
			return self.env.action_space.sample()

		else:
			# choose the action (model) that yields the biggest value:
			actions = self.predict(s)
			return np.argmax(actions)



def play_game1(model, eps, gamma, n=5):
	observation = model.env.reset()
	done = False
	totalreward = 0
	rewards = []
	states = []
	actions = []
	iters = 0
	# array of [gamma^0, gamma^1, ..., gamma^(n-1)]
	multiplier = np.array([gamma]*n)**np.arange(n)
	
	# while not done and iters < 200:
	while not done:
		# in earlier versions of gym, episode doesn't automatically
		# end when you hit 200 steps
		action = model.epsilon_greedy_action(observation, eps)

		states.append(observation)
		actions.append(action)

		prev_observation = observation
		observation, reward, done, info = model.env.step(action)

		rewards.append(reward)

		# update the model
		if len(rewards) >= n:
			# return_up_to_prediction = calculate_return_before_prediction(rewards, gamma)
			return_up_to_prediction = multiplier.dot(rewards[-n:])
			G = return_up_to_prediction + (gamma**n)*np.max(model.predict(observation)[0])
			model.update(states[-n], actions[-n], G)

		# if len(rewards) > n:
		#   rewards.pop(0)
		#   states.pop(0)
		#   actions.pop(0)
		# assert(len(rewards) <= n)

		totalreward += reward
		iters += 1

	if n==1:
		rewards = []
		states = []
		actions = []
	else:
		rewards = rewards[-n+1:]
		states = states[-n+1:]
		actions = actions[-n+1:]
	# unfortunately, new version of gym cuts you off at 200 steps
	# even if you haven't reached the goal.
	# it's not good to do this UNLESS you've reached the goal.
	# we are "really done" if position >= 0.5
	if observation[0] >= 0.5:
		# we actually made it to the goal
		# print("made it!")
		while len(rewards) > 0:
			G = multiplier[:len(rewards)].dot(rewards)
			model.update(states[0], actions[0], G)
			rewards.pop(0)
			states.pop(0)
			actions.pop(0)
	else:
		# we did not make it to the goal
		# print("didn't make it...")
		while len(rewards) > 0:
			guess_rewards = rewards + [-1]*(n - len(rewards))
			G = multiplier.dot(guess_rewards)
			model.update(states[0], actions[0], G)
			rewards.pop(0)
			states.pop(0)
			actions.pop(0)
	
	return iters, totalreward



def play_game(model, eps, gamma, n):
	# 'n' is the number of steps to consider before making an update;
	# we will need to keep track of 'n' recent rewards
	total_reward = 0
	# storages:
	rewards = []
	states = []
	actions = []
	# multiplier array of powers of gamma:     
	gammas = gamma**np.arange(n) # [gamma**0, gamma**1, ..., gamma**(n-1)]
	steps = 1
	
	s = model.env.reset()  
	done = False
	while not done:        
		# choose an action:        
		a = model.epsilon_greedy_action(s, eps)

		states.append(s)
		actions.append(a)

		# take the action:
		s_prime, r, done, _ = model.env.step(a)

		rewards.append(r)
		
		if len(rewards) >= n:
			# update the parameters of the model based pn the 'n' recent rewards:
			# G = R1 + gamma * R2 + ... + gamma**(n-1) * Rn + gamma**n * V(s_prime)
			G = gammas.dot(rewards[-n:]) + gamma**n * np.max(model.predict(s_prime)) # the estimated value of the return
			# update the values for the 'n'th recent state:
			model.update(states[-n], actions[-n], G)
		   
		# the next state becomes current:
		s = s_prime

		total_reward += r
		steps += 1

	# empty the cache, i.e., delete all the elements up to -n inclusively:
	rewards = rewards[-n+1:]
	states = states[-n+1:]
	actions = actions[-n+1:]  

	# NOTE: we must not forget to update the remaining states;
	#       consider 2 cases: 
	#           1) episode is over and we've reached the goal 
	#           2) or not
	# print('\n\nlen(rewards):', len(rewards))
	if s[0] >= 0.5:
		# print('\nwe made it')
		# we've reached the top of the heel; 
		while len(rewards) > 0:
			# update every state as before;
			# every subsequent reward is deemed 0:        
			G = gammas[:len(rewards)].dot(rewards) + 0 
			model.update(states[0], actions[0], G)
			# remove the first elements in the storages
			# to update the next states' values:
			rewards.pop(0)
			states.pop(0)
			actions.pop(0)
	else:
		# print('\nwe didn\'t make it')
		# we have to assume that we wouldn't make it in next 'n' steps either:
		while len(rewards) > 0:
			# update every state as before;
			# every subsequent reward is deemed -1:
			assumed_rewards = rewards + [-1]*(n - len(rewards))
			G = gammas.dot(assumed_rewards)
			model.update(states[0], actions[0], G)
			# remove the first elements in the storages
			# to update the next states' values:
			rewards.pop(0)
			states.pop(0)
			actions.pop(0)

	return steps, total_reward
		


def plot_running_avg(total_rewards):
	N = len(total_rewards)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean( total_rewards[max(0, t-99):t+int(t==0 or t>=100)] )
	plt.plot(running_avg)
	plt.title('Running Average')
	plt.show()



def plot_cost_to_go(model, num_tiles=20):
	''' Plots negative of the Optimal Value Function V*(s). '''
	x = np.linspace(model.env.observation_space.low[0], model.env.observation_space.high[0], num_tiles)
	y = np.linspace(model.env.observation_space.low[1], model.env.observation_space.high[1], num_tiles)
	X, Y = np.meshgrid(x, y) # each of shape (num_tiles, num_tiles)
	Z = np.apply_along_axis(lambda v: -np.max(model.predict(v)), 2, np.dstack([X, Y]))
	# print('Z.shape:', Z.shape) # (num_tiles, num_tiles)
	# exit()

	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(X, Y, Z, 
		rstride=1, cstride=1, cmap='coolwarm', edgecolors='k')#, vmin=-1.0, vmax=1.0)
	ax.set_xlabel('Position')
	ax.set_ylabel('Velocity')
	ax.set_zlabel('-V*(s)')
	ax.set_title('Cost-To-Go Function')
	fig.colorbar(surf)
	plt.show()



def main():
	env = gym.make('MountainCar-v0')
	# set max number of steps per episode:
	env._max_episode_steps = 500
	
	feature_transformer = FeatureTransformer(env)
	# print('feature.dimensionality:', feature_transformer.dimensionality)
	# s = env.reset()
	# feature = feature_transformer.transform([s])
	# print('feature.shape:', feature.shape)
	
	# if required, save the video of our Agent playing the game:
	filename = os.path.basename(__file__).split('.')[0]
	save_as = filename + ' ' + str(datetime.now()).replace(':', ' ') # enable for windows
	if 'wrap' in sys.argv:
		env = wrappers.Monitor(env, save_as)

	# instantiate our model:
	model = Model(env, feature_transformer, 'constant', 0.001)

	gamma = 0.99 # the discount factor
	n = 5 # the number of steps to consider before updating
	n_episodes = 1000
	avg_length = 0
	total_rewards = np.empty(n_episodes)
	
	for t in range(n_episodes):
		t0 = datetime.now()
		eps = 0.1*(0.97**t)
		# eps = 0 # using optimistic initial values method
		steps, total_reward = play_game(model, eps, gamma, n)

		total_rewards[t] = total_reward
		avg_length += (steps - avg_length)/(t+1)
		if t % 100 == 0:
			print('episode: %d\tETA: %s\tavg reward over last 100: %.3f\teps: %.3f' % \
				(t, datetime.now() - t0, np.mean( total_rewards[max(0, t-99):t+int(t==0 or t>=100)] ), eps))

	print('\navg episode length:', avg_length)
	print('\navg reward for last 100 episodes:', np.mean(total_rewards[-100:]))
	
	plt.plot(total_rewards)
	plt.title('Rewards')
	plt.xlabel('episodes')
	plt.show()

	plot_running_avg(total_rewards)

	plot_cost_to_go(model)



if __name__ == '__main__':
	main()