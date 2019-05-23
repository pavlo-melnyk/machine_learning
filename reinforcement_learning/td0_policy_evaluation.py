'''
An implementation of the TD(0) algorithm
for the Gridworld game (see 'gridworld.py' for reference).

NOTE: this is a solution to the Prediciton Problem, 
      i.e., a method for finding the value function given policy.
'''

import numpy as np 
import matplotlib.pyplot as plt 
from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

N_EPISODES = 2000
GAMMA = 0.9 # the discount factor
ALPHA = 0.1 # sort of a 'learning rate', a constant for the running avg
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

	# display policy:
	print('\nfixed policy:')
	print_policy(policy, grid)

	# initialize the value function:
	V = {}
	for s in grid.all_states:
		V[s] = 0

	print()

	# TD(0):
	for t in range(N_EPISODES):
		if t % 100 == 0:
			print('episode:', t)

		# play an episode
		# starting position is the same for every episode:
		s = (2, 0)
		grid.set_state(s)
		
		# NOTE: timing! we're in a state s(t), for landing in which 
		#       we've received a reward r(t) 
		# states_n_rewards = [(s, 0)]

		while not grid.game_over:
			cur_s = s # current position, s(t)
			# take an epsilon-greedy action, arrive in a state,
			# and receive a reward:
			a = random_action(policy[s]) 
			r = grid.move(a)
			s = grid.current_state # our s(t+1) = s_prime 

			# make a TD(0) update: 
			# we can do it online, b/c we're using the expected value
			# of the return, r + GAMMA*V(s_prime),
			# NOT the return itself			
			V[cur_s] = V[cur_s] + ALPHA*(r + GAMMA*V[s] - V[cur_s])

	print('\nfinal values:')
	print_values(V, grid)