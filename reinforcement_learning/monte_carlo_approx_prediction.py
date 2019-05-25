'''
An implemetnation of the First-Visit Monte Carlo Policy Evaluation
with Approximation algorithm (a solution to the Prediction Problem)
given a deterministic policy for the Gridworld game (see 'gridworld.py' for reference).

We now assume that the state-transition probabilities, p(s',r|s,a),
are random, s.t. p(a|s) = 0.5, and p(!a|s) = 0.5/3.
'''
import numpy as np 
import matplotlib.pyplot as plt 

from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_in_windy_gridworld import random_action, POLICY


ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
ALPHA = 0.001 # the learning rate
GAMMA = 0.9 # the discount factor
N_EPISODES = 20000 # the number of episodes to play


def play_game(grid, policy):
	''' Takes in our grid and policy.
	Returns a list of state-return tuples.
	'''
	# the Agent needs to be able to appear in any state,
	# but the current deterministic policy doesn't allow this

	# the starting position:
	s = (2, 0) # s(t)
	grid.set_state(s)	

	# the list of state-rewards is needed to calculate returns:
	states_and_rewards = [(s, 0)] # the reward, r(t), for landing 
	                              # in the starting state, s(t), is 0

	# play the game until it's over:
	while not grid.game_over:
		# we are in the windy gridworld - state-transitions are now random:
		a = random_action(policy[s]) # returns policy[s] with p=0.5
		# take this action and receive a reward:
		r = grid.move(a)
		s = grid.current_state # now we're in a new state
		# store the pair:
		states_and_rewards.append([s, r])

	states_and_returns = []
	G = 0 # the return of the terminal state is 0
	# we should not add it to our list,
	# it is the first in the reversed order:
	first = True
	for s, r in reversed(states_and_rewards):
		if first:
			first = False
		else:
			states_and_returns.append((s, G))

		# calculate the return by definition:
		G = r + GAMMA * G

	# return the state-return pairs in the chronological order:
	states_and_returns.reverse()	
	return states_and_returns


def feature_transformer(s):
	''' Transforms a state, s, into a feature vector, x.'''
	x1, x2 = s 
	return np.float64([1, x1 - 1, x2 - 1.5, x1 * x2 - 3])
	# return np.float64([1, x1**2, x2**2, x1*x2])


if __name__ == '__main__':
	grid_type = input('\nchoose grid type (\'standard\', \'negative\'):\n')

	if grid_type == 'negative':
		step_cost = float(input('\nenter step_cost (e.g. \'-1\' or \'-0.1\'):\n').strip())
		# get the grid:
		grid = negative_grid(step_cost=step_cost)

	else:
		# assuming the standard grid:
		grid = standard_grid()

	# display rewards:
	print('\nrewards:')
	print_values(grid.rewards, grid)

	# the policy is deterministic:
	policy = POLICY
	print('\nfixed policy:')
	print_policy(policy, grid)

	# number of features:
	D = 4
	
	# randomly initialize the parameters of our linear model:
	theta = np.random.randn(D) / np.sqrt(D)

	print()
	
	deltas = [] # for convergence check
	t = 1.0 # learning rate divisor

	################### First-Visit Monte Carlo with Approximation: ###################
	for i in range(N_EPISODES):
		if i % 100 == 0:
			t += 0.1
		if i % 1000 == 0:
			print('episode:', i)

		# decrease the learning rate:
		alpha = ALPHA / t

		# play the game using our fixed policy:
		states_and_returns = play_game(grid, policy)
		seen_states = set()

		# Gradient Descent:
		max_change = 0
		for s, G in states_and_returns:
			if s not in seen_states:
				old_theta = theta.copy()
				# transform the state into a feature vector:
				x = feature_transformer(s)
				# get the prediction with our linear model:
				v_hat = theta.T.dot(x) 
				# update the parameters:
				theta += alpha * x.dot((G - v_hat))
				
				max_change = max(max_change, np.abs(theta - old_theta).sum())
				seen_states.add(s)

		deltas.append(max_change)

	plt.plot(deltas)
	plt.title('theta convergence')
	plt.xlabel('episodes')
	plt.show()

	# the estimate value function:
	V_hat = {}
	print('\nstate  --->  feature vector')
	for s in policy.keys():
		x = feature_transformer(s)
		print(s, '--->', x)
		V_hat[s] = theta.T.dot(x)

	# values:
	print('\nestimated values:')
	print_values(V_hat, grid)

	print('\ntheta:\n', theta)