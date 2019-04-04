'''
An implementation of the Value Iteration algorithm -
yet another way to solve the control problem.
It is similar to Policy Iteration. 
The key difference is that Value Iteration combines two steps together:
policy evaluation and policy improvement.

NOTE: the policy in this script is considered deterministic, 
	  i.e., all state-transitions are NOT random.
'''

import numpy as np 
import matplotlib.pyplot as plt 

from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy


THRESHOLD = 1e-4 # a small value to check the convergence of the value function
GAMMA = 0.9 # the discount factor
ALL_POSSIBLE_ACTIONS = ['U', 'R', 'D', 'L']


def main(grid_type='negative'):
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

	states = grid.all_states

	# STEP 1: randomly initialize the value function, V(s):
	V = {} # the values
	for s in states:
		# as an option, initialize to 0:
		# V[s] = 0 

		# check if not a terminal state:
		if s in grid.actions:
			V[s] = np.random.random()
		else:
			V[s] = 0

	print('\ninitial values:')
	print_values(V, grid)


	# STEP 2: value iteration
	while True:
		max_change = 0 

		for s in states:
			old_v = V[s]

			# if we're not in a terminal state:
			if s in grid.actions:
				# choose an action that results in the maximum value
				# for this state:
				best_v = np.float('-inf')
				# best_a = np.random.choice(ALL_POSSIBLE_ACTIONS)

				for a in ALL_POSSIBLE_ACTIONS:
					# arrive in the state:
					grid.set_state(s)

					# take the action and receive the reward:
					r = grid.move(a)

					# calculate the Bellman equation:
					v = r + GAMMA * V[grid.current_state]

					if v > best_v:
						best_v = v
						# p[s] = a      # we'll do it in another loop later

				# update the value of this state:
				V[s] = best_v

				# update the maximum change:
				max_change = max(max_change, np.abs(old_v - V[s]))

		# check if converged:
		if max_change < THRESHOLD:
			break


	# STEP 3: take our optimal value funciton
	#         and find our optimal policy
	p = {} # the policy
	for s in states:
		best_a = None
		best_v = float('-inf')

		# if not a terminal state:
		if s in grid.actions:
			# find the best action:
			for a in ALL_POSSIBLE_ACTIONS:
				grid.set_state(s)
				r = grid.move(a)
				v = r + GAMMA * V[grid.current_state]

				if v > best_v:
					best_v = v
					best_a = a

			p[s] = best_a 


	# optimal values:
	print('\noptimal values:')
	print_values(V, grid)

	# optimal policy:
	print('\noptimal policy:')
	print_policy(p, grid)




if __name__ == '__main__':
	grid_type = input('\nchoose grid type (\'standard\', \'negative\'):\n')
	main(grid_type)