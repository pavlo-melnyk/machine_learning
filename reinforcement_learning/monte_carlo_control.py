'''
An implementation of the First-Visit Monte Carlo without Exploring Starts algorithm
for the Gridworld game (see 'gridworld.py' for reference).

NOTE: this is a solution to the Control Problem, i.e., a method for finding
      the optimal policy.

'''
import numpy as np 
import matplotlib.pyplot as plt

from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy


ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
GAMMA = 0.9 # the discount factor
N_EPISODES = 5000 # the number of episodes to play


def random_action(a, eps=0.1):
	''' Epsilon-Soft.
	Returns given action, a, with the probability
	p < (1 - eps + eps/len(ALL_POSSIBLE_ACTIONS));

	Returns any other action, !a, with the probability 
	eps/len(ALL_POSSIBLE_ACTIONS).
	'''

	p = np.random.random()
	# if p < (1 - eps + eps/len(ALL_POSSIBLE_ACTIONS)):
	# 	return a 
	# else:
	# 	tmp = list(ALL_POSSIBLE_ACTIONS)
	# 	tmp.remove(a)
	# 	return np.random.choice(tmp)

	# or equivalently:
	if p < (1 - eps):
		return a 
	else:
		return np.random.choice(ALL_POSSIBLE_ACTIONS)



def play_game(grid, policy):
	''' Takes in grid and policy.
	Returns a list of state-action-return triples.
	'''
	
	# starting position is the same for every episode:
	s = (2, 0) # starting state
	grid.set_state(s)	

	# do Epsilon-Soft:
	a = random_action(policy[s]) 

	# define our (s(t), a(t), r(t)) triple:
	# NOTE: timing! we received reward r(t) for taking action a(t-1) and 
	#       landing in state s(t), from where we're now taking action a(t)
	states_actions_rewards = [(s, a, 0)]

	while True:
		# print('I am in the state', s)
		# print('taking action', a)
		r = grid.move(a)
		s = grid.current_state
		# print('arriving in the state', s)
		
		if grid.game_over:
			# print('\t-----> episode is over')
			states_actions_rewards.append((s, None, r))
			break

		# else:
		# do Epsilon-Soft:
		a = random_action(policy[s])
		states_actions_rewards.append((s, a, r))
	
	G = 0 # the return of the terminal state
	states_actions_returns = []
	# we don't want do include the return of the
	# terminal state to the states_actions_returns list:
	first = True
	for s, a, r in reversed(states_actions_rewards):
		if first:
			first = False
		else:
			states_actions_returns.append((s, a, G))

		# calculate the return by definition:
		G = r + GAMMA * G

	states_actions_returns.reverse()
	return states_actions_returns


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

	# initialize the policy, action-value function, and number of visits per state given an action:
	policy = {}
	Q = {}
	N = {}
	returns = {}
	for s in states:
		if s in grid.actions:
			policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
			Q[s] = {}
			N[s] = {}
			for a in ALL_POSSIBLE_ACTIONS:
				Q[s][a] = 0
				N[s][a] = 0
				returns[(s,a)] = []

	# display the initial policy:
	print('\ninitial policy:')
	print_policy(policy, grid)

	# First-Visit Monte Carlo Exploring-Starts:
	deltas = []
	for i in range(N_EPISODES):
		if i % 1000 == 0:
			print('\nepisode:', i)

		# generate an episode:
		max_change = 0 # to check convergence of the value function
		states_actions_returns = play_game(grid, policy)
		visited_s_and_a_pairs = set()

		# STEP 1: Policy Evaluation
		# print('\n------> Performing Policy Evaluation...')
		for s, a, G in states_actions_returns:
			if (s,a) not in visited_s_and_a_pairs:
				old_q = Q[s][a]
				# returns[(s,a)].append(G)
				# Q[s][a] = np.mean(returns[(s,a)])

				# calculate the running average:
				N[s][a] += 1
				Q[s][a] = (1 - 1 / N[s][a]) * Q[s][a] + (1 / N[s][a]) * G
				
				max_change = max(max_change, np.abs(old_q - Q[s][a]))
				visited_s_and_a_pairs.add((s,a))

		deltas.append(max_change)

		# STEP 2: Policy Improvement
		# print('\n------> Performing Policy Improvement...')
		for s in policy.keys():
			max_q = np.float('-inf')
			best_a = None
			for a in ALL_POSSIBLE_ACTIONS:
				if Q[s][a] > max_q:
					max_q = Q[s][a]
					best_a = a
			policy[s] = best_a
			
	V = {} # state-value function
	# do argmax on Q(s,a):
	for s in policy.keys():
		max_q = np.float('-inf')
		for a in ALL_POSSIBLE_ACTIONS:
			if Q[s][a] > max_q:
				max_q = Q[s][a]
		V[s] = max_q


	# print policy:
	print('\nfinal policy:')
	print_policy(policy, grid)

	# print values:
	print('\nfinal values:')
	print_values(V, grid)
	
	# plot the deltas:
	plt.plot(deltas)
	plt.title('Q(s,a) Convergence')
	plt.xlabel('episode')
	plt.ylabel('max change')
	plt.show()
	