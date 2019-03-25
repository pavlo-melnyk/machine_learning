import numpy as np 

from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values


THRESHOLD = 1e-4 # to check whether policy's not changed
GAMMA = 0.9 # the discount factor
ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']


def main(grid_type='negative'):
	# NOTE: every p(s',r|s,a) is deterministic (1 or 0)
	if grid_type == 'negative':
		# get the grid:
		grid = negative_grid()

	else:
		# assuming the standard grid:
		grid = standard_grid()

	# print the rewards:
	print('\nrewards:')
	print_values(grid.rewards, grid) # prints any dict with
	                                 # a tuple of numbers as the key
	                                 # and a number as the value


	# STEP 1: randomly initialize V(s) and the policy, pi(s):
	V = {}
	states = grid.all_states
	for s in states:
		# we can simply initialize all to zero:
		V[s] = 0
		# or perform a random initialization:
		# if s in grid.actions: # if not a terminal state
		# 	V[s] = np.random.random()
		# else:
		# 	# terminal
		# 	V[s] = 0
	print('\ninitial values:')
	print_values(V, grid)


	policy = {}
	for s in grid.actions.keys():
		policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
	print('\ninitial policy:')
	print_policy(policy, grid)


	# STEP 2: alternate between policy evaluation and policy improvement:
	# repeat untill convergence:
	i = 0
	while True:

		# STEP 2A: iterative policy evaluation
		while True:
			# NOTE: all of the actions, next states and rewards 
			#       are considered deterministic

			max_change = 0
			for s in states:
				old_v = V[s] # save the old value of the state

				# check if not a terminal state:
				if s in grid.actions:
					grid.set_state(s)

					# take an action according to the policy and get the reward:
					a = policy[s]
					r = grid.move(a)

					# the "look-ahead" - get the value of the next state, s_prime:
					s_prime = grid.current_state
					# s_prime is needed in order to calculate 
					# the value of the current state - the Bellman equation:
					V[s] = r + GAMMA * V[s_prime]

					# update max_change:
					max_change = max(max_change, np.abs(V[s] - old_v))

			# check if converged:
			if max_change < THRESHOLD:
				break

		# STEP 2B: policy iteration
		# for each state we take an action according to the policy
		# and check whether there is a better action - take all possible
		# actions from that state and calculate the values; 
		# we choose the action that results in the max value of the state.
		policy_improved = False
		for s in states:

			# check if not a terminal-state:
			if s in grid.actions:					
				grid.set_state(s) # yep, don't forget to set the position!

				# save the old policy:
				old_a = policy[s]
				
				max_v = np.float('-inf') # worse is unlikely to occur

				# choose the best action among all the possible ones:			
				for a in ALL_POSSIBLE_ACTIONS:	
					# print('reached here!')	
					grid.set_state(s)

					# take an action, receive your keto-chocolate bar:
					r = grid.move(a)

					s_prime = grid.current_state
					new_v = r + GAMMA * V[s_prime]

					# compare the values:
					if new_v > max_v:
						max_v = new_v
						better_a = a
						# change the policy:
						policy[s] = better_a 
						

				if old_a != better_a:
					# print('policy_improved')
					policy_improved = True

		# if policy has changed, we need to recalculate the values of all states -
		# get back to STEP 2A; 
		# else - we're done! 
		# and since the policy's not changed, the values remain the same:
		if not policy_improved:
			break

		i += 1

	print('\niterations to converge:', i)

	# print the values:
	print('\nvalues:')
	print_values(V, grid)

	# print the policy:
	print('\nthe improved policy:')
	print_policy(policy, grid)




if __name__ == '__main__':
	grid_type = input('\nchoose grid type (\'standard\', \'negative\'):\n')
	
	main(grid_type)