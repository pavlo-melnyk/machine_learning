'''
An implementation of the TD(0) Semi-Gradient algorithm
for the Gridworld game (see 'gridworld.py' for reference).

NOTE: this is a solution to the Prediciton Problem, 
      i.e., a method for finding the value function given 
      a deterministic policy.
'''

import numpy as np 
import matplotlib.pyplot as plt 
from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy


N_EPISODES = 20000
GAMMA = 0.9 # the discount factor
ALPHA = 0.001 # the learning rate
ALL_POSSIBLE_ACTIONS = ['U', 'R', 'D', 'L']


def random_action(a, eps=0.1):
	''' Epsilon-Soft. 
	Used in order to ensure all states are visited.
	
	Returns the given action, a, with the probability
	p < (1 - eps + eps/len(ALL_POSSIBLE_ACTIONS));
	
	Returns any other action, !a, with the probability 
	eps/len(ALL_POSSIBLE_ACTIONS).
	'''
	p = np.random.random()
	if p > eps:
		return a
	else:
		return np.random.choice(ALL_POSSIBLE_ACTIONS)


def feature_transformer(s):
	''' Transforms a state, s, into a feature vector, x.'''
	x1, x2 = s 
	return np.float64([1, x1 - 1, x2 - 1.5, x1 * x2 - 3])


if __name__ == '__main__':
	grid_type = input('\nchoose grid type (standard/negative):\n').strip()

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

	# consider a deterministic policy:
	# (the same as in iterative_policy_evaluation.py)
	policy = {
		(0, 0): 'R',
		(0, 1): 'R',
		(0, 2): 'R',
		(1, 0): 'U',
		(1, 2): 'R',
		(2, 0): 'U',
		(2, 1): 'R',
		(2, 2): 'R',
		(2, 3): 'U',
	}

	# display our policy:
	print('\nfixed policy:')
	print_policy(policy, grid)

	# number of features:
	D = 4

	# randomly initialize the parameters of our model:
	theta = np.random.randn(D) / np.sqrt(D)

	print()

	############################# TD(0) Semi-Gradient: #############################
	deltas = [] # to check the convergence of theta
	t = 1.0 # the learning rate divisor
	for i in range(N_EPISODES):
		if i % 10 == 0:
			t += 0.01
		if i % 1000 == 0:
			print('episode:', i)

		alpha = ALPHA / t # decay the learning rate
		max_change = 0 # to check the convergence

		# play an episode
		# starting position is the same for every episode:
		s = (2, 0)
		grid.set_state(s)

		# NOTE: timing! we're in a state s(t), for landing in which 
		#       we've received a reward r(t) and from where we take 
		#       an action a(t) and receive a reward r(t+1)

		while not grid.game_over:
			old_theta = theta.copy()

			cur_s = s # current position, s(t)
			# transform the state to a feture vector:
			cur_x = feature_transformer(s)
			# get the prediction for the current state, V_hat(s):		
			cur_v_hat = theta.T.dot(cur_x)

			# take an epsilon-greedy action:
			a = random_action(policy[cur_s]) # a(t)
			r = grid.move(a) # r(t+1)
			s = grid.current_state # our s(t+1) = s_prime 

			# make a TD(0) update: 
			# b/c we're using the expected value of the return
			# as the target,  r + GAMMA*V(s_prime), NOT the return itself, 
			# so in order to make an update, we also need to get 
			# the model output for the state s(t+1):
			if grid.is_terminal(s):
				# the value of the terminal state is 0:
				target = r
			else:
				x = feature_transformer(s)
				v_hat = theta.T.dot(x)
				target = r + GAMMA*v_hat

			# semi-gradient update:
			theta = theta + alpha*(target - cur_v_hat)*cur_x

			max_change = max(max_change, np.abs(old_theta - theta).sum())

		deltas.append(max_change)


	plt.plot(deltas)
	plt.title('theta convergence')
	plt.xlabel('episodes')
	plt.show()

	# estimate the value function:
	V_hat = {}
	# print('\nstate  --->  feature vector')
	for s in policy.keys():
		x = feature_transformer(s)
		# print(s, '--->', x)
		V_hat[s] = theta.T.dot(x)

	# values:
	print('\nestimated values:')
	print_values(V_hat, grid)

	print('\ntheta:\n', theta)


