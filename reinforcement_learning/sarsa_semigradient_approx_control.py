'''
An implementation of the SARSA Semi-Gradient algorithm
for the Gridworld game (see 'gridworld.py' for reference).

NOTE: this is a solution to the Control Problem, 
	  i.e., a method for finding the optimal policy.
'''

import numpy as np 
import matplotlib.pyplot as plt 
from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy


N_EPISODES = 20000
GAMMA = 0.9 # the discount factor
ALPHA = 0.1 # the learning rate
ALL_POSSIBLE_ACTIONS = ['U', 'R', 'D', 'L']

# for one-hot encoding
SA2IDX = {} # state_and_action ---> index
IDX = 0


def random_action(a, eps=0.5):
	'''Epsilon-Soft. 
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


def feature_transformer(s, a):
	''' Transforms a state and action pair, (s, a), into a feature vector, x.'''
	# TODO: create better features
	x1, x2 = s 
	x = np.float64([
		1                                    ,
		x1 - 1             if a == 'U' else 0,
		(x2 - 1.5)/1.5     if a == 'U' else 0,
		# (x1*x1 - 2)/2      if a == 'U' else 0,
		(x1*x2 - 3)/3      if a == 'U' else 0,
		# (x2*x2 - 4.5)/4.5  if a == 'U' else 0,
		1                  if a == 'U' else 0,
		x1 - 1             if a == 'D' else 0,
		(x2 - 1.5)/1.5     if a == 'D' else 0,
		# (x1*x1 - 2)/2      if a == 'D' else 0,
		(x1*x2 - 3)/3      if a == 'D' else 0,
		# (x2*x2 - 4.5)/4.5  if a == 'D' else 0,
		1                  if a == 'D' else 0,
		x1 - 1             if a == 'L' else 0,
		(x2 - 1.5)/1.5     if a == 'L' else 0,
		# (x1*x1 - 2)/2      if a == 'L' else 0,
		(x1*x2 - 3)/3      if a == 'L' else 0,
		# (x2*x2 - 4.5)/4.5  if a == 'L' else 0,
		1                  if a == 'L' else 0,
		x1 - 1             if a == 'R' else 0,
		(x2 - 1.5)/1.5     if a == 'R' else 0,
		# (x1*x1 - 2)/2      if a == 'R' else 0,
		(x1*x2 - 3)/3      if a == 'R' else 0,
		# (x2*x2 - 4.5)/4.5  if a == 'R' else 0,
		1                  if a == 'R' else 0,
	])

		
	# NOTE: we won't use one-hot encoding, b/c we would 
	#       have the same number of parameters to store
	#       as with the tabular method we used before;
	#       however, it helps debug the code - if we 
	#       use it and it yields the 'correct' result
	#       it means that the code works and we should
	#       carefully engineer the features for our
	#       linear model to correctly approximate the 
	#       value function.

	# uncomment to do one-hot encoding:
	# x = np.zeros(IDX+1) # len(x) = |S|*|A|
	# idx = SA2IDX[s][a]
	# x[idx] = 1.0 
	return x


def best_value_and_action(s):
	''' Predicts values for a given state
	for all actions. Returns the maximum value
	and the corresponding action.
	'''
	max_q = np.float('-inf')
	best_a = None
	for a in ALL_POSSIBLE_ACTIONS:
		q_sa = theta.T.dot(feature_transformer(s, a))
		if q_sa > max_q:
			max_q = q_sa
			best_a = a
	return max_q, best_a


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

	# initialize the count for s:
	N = {}
	for s in grid.all_states:
		N[s] = 0

	# populate the dictionary needed for one-hot encoding:
	for s in grid.actions.keys():
		SA2IDX[s] = {}
		for a in ALL_POSSIBLE_ACTIONS:
			SA2IDX[s][a] = IDX
			IDX += 1

	# print('IDX:', IDX)
	# print('SA2IDX:', SA2IDX)
	# exit()

	# number of features:
	D = len(feature_transformer((0,0), 'U'))

	# randomly initialize the parameters of our linear model:
	theta = np.random.randn(D) / np.sqrt(D)

	print()

	############################# SARSA Semi-Gradient: #############################
	deltas = [] # to check the convergence of the theta
	t = 1.0 # the divisor for epsilon in epsilon-greedy	
	t2 = 1.0 # the divisor for the learning rate

	# repeat until convergence:
	for i in range(N_EPISODES):
		# adaptive epsilon for epsilon-greedy:
		if i % 100 == 0:
			t += 1e-2
			t2 += 1e-2
		if i % 1000 == 0:
			print('episode:', i)

		# decay the learing rate:
		alpha = ALPHA / t2
		
		# starting position is the same for every episode:
		s = (2, 0)
		grid.set_state(s)

		# choose an epsilon-greedy action, a(t): 
		_, a = best_value_and_action(s)
		a = random_action(a, eps=0.5/t)

		# get a feature vector, x, given the (s,a) pair:
		x = feature_transformer(s, a)

		# calculate the approximation, q_hat,
		# i.e., the prediction for the (s,a) pair, Q_hat(s, a):
		q_hat = theta.T.dot(x)

		# NOTE: timing! we're in a state s(t), for landing in which 
		#       we've received a reward r(t) and from where we take 
		#       an action a(t) and receive a reward r(t+1) 

		max_change = 0 # to check the convergence of theta
		
		while not grid.game_over:
			cur_s = s # s(t)
			cur_a = a # a(t)
			cur_x = x # the gradient in the update equation
			cur_q_hat = q_hat
			old_theta = theta.copy() 

			N[s] += 1

			# take the epsilon-greedy action, a(t), arrive in a state,
			# and receive a reward:			
			r = grid.move(a) # r(t+1) - the reward for landing in s(t+1)
			s = grid.current_state # s(t+1) = s_prime

			# recall, target = r + GAMMA*q_hat(s_prime, a_prime)
			if grid.is_terminal(s):
				# the value of the terminal state is 0:
				target = r
			else:
				# since the update equation involves Q(s(t+1), a(t+1)), 
				# i.e., q_hat(s_prime, a_prime),
				# we also need to know a(t+1):
				a = random_action(best_value_and_action(s)[1], eps=0.5/t)
				x = feature_transformer(s, a)
				q_hat = theta.T.dot(x)
				target = r + GAMMA*q_hat			

			# make a semi-gradient update: 
			theta += alpha*(target - cur_q_hat)*cur_x 

			max_change = max(max_change, np.abs(old_theta - theta).sum())

		deltas.append(max_change)


	plt.plot(deltas)
	plt.xlabel('episodes')
	plt.ylabel('max change')
	plt.title('theta convergence')
	plt.show()

	# find the optimal value function, V*(s),
	# and the optimal policy:
	V = {}
	policy = {}

	# do argmax on Q*(s,a):
	for s in grid.actions.keys():
		V[s], policy[s] = best_value_and_action(s)
	
	print('\noptimal values:')
	print_values(V, grid)

	# display the policy:
	print('\noptimal policy:')
	print_policy(policy, grid)

	# the parameters of our linear model:
	print('\ntheta:\n', theta)

	# for debugging:
	total_evaluations = np.sum(list(N.values()))
	for s, n in N.items():
		N[s] = n / total_evaluations

	print('\nproportion of total evaluation time per state:')
	print_values(N, grid)	