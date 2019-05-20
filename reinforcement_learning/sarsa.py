'''
An implementation of the SARSA algorithm
for the Gridworld game (see 'gridworld.py' for reference).

NOTE: this is a solution to the Control Problem, 
	  i.e., a method for finding the optimal policy.
'''

import numpy as np 
import matplotlib.pyplot as plt 
from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

N_EPISODES = 10000
GAMMA = 0.9 # the discount factor
ALPHA = 0.1 # sort of a 'learning rate', a constant for the running avg
ALL_POSSIBLE_ACTIONS = ['U', 'R', 'D', 'L']


def best_value_and_action(Q, s):
	''' Does argmax on Q(s,a).'''
	max_q = np.float('-inf')
	best_a = None
	for a in ALL_POSSIBLE_ACTIONS:
		if Q[s][a] > max_q:
			max_q = Q[s][a]
			best_a = a
	return max_q, best_a


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

	# initialize the action-value function and the counts for s and (s, a) pairs:
	Q = {}
	N = {}
	alpha_divisor = {}
	for s in grid.all_states:
		Q[s] = {}
		N[s] = 0
		alpha_divisor[s] = {}
		for a in ALL_POSSIBLE_ACTIONS:
			Q[s][a] = 0
			alpha_divisor[s][a] = 1.0

	print()

	# SARSA:
	deltas = [] # to check the convergence of the value function
	t = 1.0 # the divisor for epsilon in epsilon-greedy	
	for i in range(N_EPISODES):
		# adaptive epsilon for epsilon-greedy:
		if i % 100 == 0:
			t += 1e-2
		if i % 500 == 0:
			print('episode:', i)
		
		# starting position is the same for every episode:
		s = (2, 0)
		grid.set_state(s)
		a = random_action(best_value_and_action(Q, s)[1], eps=0.5/t)

		# NOTE: timing! we're in a state s(t), for landing in which 
		#       we've received a reward r(t) 
		# states_n_rewards = [(s, 0)]
		max_change = 0 
		
		while not grid.game_over:
			cur_s = s # s(t)
			cur_a = a # a(t)
			old_q = Q[cur_s][cur_a] # Q(s(t), a(t))
			N[s] += 1			
			
			# take an epsilon-greedy action, arrive in a state,
			# and receive a reward:			
			r = grid.move(a) # r(t+1) - the reward for landing in s(t+1)
			s = grid.current_state # s(t+1) = s_prime
			# we also need to know a(t+1), since the update equation
			# involves Q(s(t+1), a(t+1)):
			a = random_action(best_value_and_action(Q, s)[1], eps=0.5/t)

			# decay the learing rate based on the number of visits per state given an action:
			alpha = ALPHA/alpha_divisor[cur_s][cur_a]
			alpha_divisor[cur_s][cur_a] += 0.005

			# make a TD(0) update: 
			Q[cur_s][cur_a] = Q[cur_s][cur_a] + alpha*(r + GAMMA*Q[s][a] - Q[cur_s][cur_a]) 

			max_change = max(max_change, np.abs(old_q - Q[cur_s][cur_a]))
		deltas.append(max_change)

	# find the optimal value function, V(s),
	# and the optimal policy:
	V = {}
	policy = {}

	# do argmax on Q(s,a):
	for s in grid.actions.keys():
		V[s], policy[s] = best_value_and_action(Q, s)
	
	print('\noptimal values:')
	print_values(V, grid)

	# display the policy:
	print('\noptimal policy:')
	print_policy(policy, grid)

	# for debugging:
	total_n_evaluations = np.sum(list(N.values()))
	for s, n in N.items():
		N[s] = n / total_n_evaluations

	print('\nproportion of total evaluation time per state:')
	print_values(N, grid)

	plt.plot(deltas)
	plt.xlabel('episodes')
	plt.ylabel('max change')
	plt.title('Q(s,a) convergence')
	plt.show()
