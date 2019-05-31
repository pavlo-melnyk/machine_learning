import gym
import numpy as np
import matplotlib.pyplot as plt 


def select_action(w, obs):
	return 1 if w.dot(obs) > 0 else 0


def play_game(env, w, display=False):
	''' Plays an episode. Returns the number of steps. '''
	# go to the start state:
	observation = env.reset()	
	done = False
	i = 0 # number of steps per episode
	while not done:
		if display:
			# watch the episode:
			env.render()
		# select and take an action:
		action = select_action(w, observation)			
		observation, reward, done, _ = env.step(action)
		i += 1
	return i


def random_search(env, epochs=100, n_episodes=100, verbose=False):
	''' Does a random search in the parameter space for a linear model.
	'''
	observation = env.reset()

	best_weights = None
	best_length = 0

	lengths = []
	# random search:
	for e in range(epochs):
		# ramdomly initialize the weights:
		weights = np.random.randn(*observation.shape)
		avg_length = 0

		# play the game:
		for t in range(n_episodes):
			i = play_game(env, weights)			
			# update the average:
			avg_length += (i - avg_length)/(t+1)
		lengths.append(avg_length)

		# update the weights:
		if avg_length > best_length:
			best_length = avg_length
			best_weights = weights

		if verbose and e % 5 == 0:
			print('epoch:', e, '\tavg length:', avg_length)

	return lengths, best_weights		


if __name__ == '__main__':
	# get the environment:
	env = gym.make('CartPole-v1')

	# parameters for the random search:
	epochs = 100 # number of times to adjust the weights
	n_episodes = 1000 # we play to decide whether to update the weights

	# perform the random search:
	lengths, best_weights = random_search(env, epochs, n_episodes, verbose=True)

	print('\nbest avg length:', max(lengths))
	print('\nbest weights:', best_weights)

	plt.plot(lengths)
	plt.title('Average Episode Length')
	plt.xlabel('epochs')
	plt.show()

	# play again with the best found weights:
	avg_length = 0
	for t in range(n_episodes):
		# display only the last episode;
		i = play_game(env, best_weights, t==(n_episodes-1))
		avg_length += (i - avg_length)/(t+1)

	env.close()

	print('\nfinal avg length:', avg_length)

