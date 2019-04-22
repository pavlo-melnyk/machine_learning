'''
An implemetnation of the First-Visit Monte Carlo Policy Evaluation algorithm
(a solution to the Prediction Problem) given a deterministic policy.
'''
import numpy as np 
import matplotlib.pyplot as plt 

from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from datetime import datetime


THRESHOLD = 1e-4
GAMMA = 0.9 # the discount factor
ALL_POSSIBLE_ACTIONS = ['U', 'R', 'D', 'L']


def play_game(grid, policy):
	''' Takes in our grid and policy.
	Returns a list of state-return tuples.
	'''
	# the Agent needs to be able to appear in any state,
	# but the current deterministic policy doesn't allow this

	# so we randomly select a starting state for every episode:
	states = list(grid.actions.keys()) # all states except terminal
	start_idx = np.random.choice(len(states))
	s = states[start_idx]
	grid.set_state(s)

	# the list of state-rewards is needed to calculate returns:
	states_and_rewards = [(s, 0)] # the reward for the starting state is 0

	# play the game until it's over:
	while not grid.game_over:
		# choose an action in accordance with the policy:
		a = policy[s]
		# take the action and receive a reward:
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

	# value function:
	V = {}

	# returns:
	returns = {}

	# number of visits for each state:
	N = {}

	for s in states:
		V[s] = 0
		returns[s] = []
		N[s] = 0

	# the policy is deterministic:
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

	# First-Visit Monte Carlo:
	t0 = datetime.now()
	for t in range(1000):
		states_and_returns = play_game(grid, policy)
		seen_states = set()
		for s, G in states_and_returns:
			if s not in seen_states:
				# store all the returns, calculate the mean:
				# returns[s].append(G)
				# V[s] = np.mean(returns[s])
				

				# or without storing all the returns:
				# NOTE: not as accurate, since we might not visit a state in an
				#       episode, but we always use the episode counter, t, to 
				#       calculate the running mean of the value for each state:
				# V[s] = (1 / (t + 1)) * G + (1 - 1 / (t + 1)) * V[s]

				# or keeping truck of the number of visits for each state
				# without storing all the returns:
				N[s] += 1
				V[s] = (1 / (N[s])) * G + (1 - 1 / (N[s])) * V[s]

				seen_states.add(s)

	dt = datetime.now() - t0

	# values:
	print('\nvalues:')
	print_values(V, grid)

	# return:
	print('\npolicy:')
	print_policy(policy, grid)

	print('\nETA:', dt)



if __name__ == '__main__':
	grid_type = input('\nchoose grid type (\'standard\', \'negative\'):\n')
	main(grid_type)



