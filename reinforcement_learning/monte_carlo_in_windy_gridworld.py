'''
An implementation of the First-Visit Monte Carlo Policy Evaluation algorithm for
the Gridworld game (see 'gridworld.py' for reference).

NOTE: this is a solution to the Prediction Problem, not the Control Problem.

We now assume that the state-transition probabilities, p(s',r|s,a),
are random, s.t. p(a|s) = 0.5,
and p(!a|s) = 0.5/3.
The policy is considered deterministic.

The key idea to remember:
the Agent is trying to maximize the total reward received, 
and NOT to reach whatever winning state we define.
'''
import numpy as np 

from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy


ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
GAMMA = 0.9 # the discount factor
POLICY = {
	(0, 0): 'R',
	(0, 1): 'R',
	(0, 2): 'R',
	(1, 0): 'U',
	(1, 2): 'U',
	(2, 0): 'U',
	(2, 1): 'L',
	(2, 2): 'U',
	(2, 3): 'L',
}


def random_action(a):
	# returns given action, a, with the probability 0.5
	# returns any other action, !a, with the probability 0.5/3
	p = np.random.random()
	if p < 0.5:
		return a
	else:
		tmp = list(ALL_POSSIBLE_ACTIONS)
		tmp.remove(a)
		return np.random.choice(tmp)


def play_game(grid, policy):
	''' Takes in grid and policy.
	Returns a list of state-return tuples.
	'''
	# randomly select a starting state for every episode:
	states = list(grid.actions.keys())
	start_idx = np.random.choice(len(states))
	s = states[start_idx] # starting state
	grid.set_state(s)

	states_and_rewards = [(s, 0)]
	while not grid.game_over:
		# since we are in a windy gridworld,
		# the action might be randomly changed:
		a = random_action(policy[s])
		r = grid.move(a)
		s = grid.current_state
		states_and_rewards.append((s, r))

	G = 0 # the return of the terminal state
	states_and_returns = []
	# we don't want do include the return of the
	# terminal state to the states_and_returns list:
	first = True
	for s, r in reversed(states_and_rewards):
		if first:
			first = False
		else:
			states_and_returns.append((s, G))

		# calculate the return by definition:
		G = r + GAMMA * G

	states_and_returns.reverse()
	return states_and_returns


if __name__ == '__main__':
	grid_type = input('\nchoose grid type (standard/negative):\n')

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

	# initialize value function and numper of visits per state:
	V = {}
	N = {}
	for s in states:
		V[s] = 0
		N[s] = 0

	# First-Visit Monte Carlo:
	for i in range(10000):
		states_and_returns = play_game(grid, POLICY)
		visited_s = set()
		for s, G in states_and_returns:
			if s not in visited_s:
				N[s] += 1
				V[s] = (1 - 1 / N[s]) * V[s] + (1 / N[s]) * G

	# print values:
	print('\nvalues:')
	print_values(V, grid)

	# print policy:
	print('\npolicy:')
	print_policy(POLICY, grid)










