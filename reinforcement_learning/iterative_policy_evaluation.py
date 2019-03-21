import numpy as np 
import matplotlib.pyplot as plt 
from gridworld import standard_grid


THRESHOLD = 1e-4


def print_values(V, g):
	''' Takes in a value dictionary, V, and a grid, g.
	Draws the grid, and in each position it prints the value.
	'''
	for i in range(g.width):
		print('------------------------')
		for j in range(g.height):
			v = V.get((i, j), 0)
			if v >= 0:
				print(' %.2f|' % v, end='')
			else:
				print('%.2f|' % v, end='')
		print()


def print_policy(P, g):
	''' Takes in a policy dictionary, P, and a grid, g.
	Draws the grid, and in each position it prints the action.
	Works only for deterministic policies, i.e., p(s',r| s,a) = const.
	Since we can't print more than one thing per location.
	'''
	for i in range(g.width):
		print('----------------')
		for j in range(g.height):
			a = P.get((i, j), ' ')
			print(' %s |' % a, end='')
		print()



def main(policy='uniform'):
	# let's find value function V(s), given a policy p(a|s).
	#
	# recall that there are 2 different policies:
	# 1) completely random policy;
	# 2) completely deterministic (fixed) policy.
	# we are going to find value function for both.
	#
	# NOTE: 
	# there are 2 probability distributions in the Bellman equation:
	# 1) p(a|s) - the policy, defines what action to take given the state;
	# 2) p(s',r|s,a) - state-transition probability, 
	#                  defines the next state and reward
	#                  given a state-action pair.
	# we will only model a uniform random policy, i.e., p(a|s) = uniform.
	grid = standard_grid()

	# the states will be positions (i, j).
	# gridworld is simpler than tic-tac-toe, b/c there's only one player
	# (i.e., a robot) that can only be at one position at a time.
	states = grid.all_states


	if policy == 'uniform':
		#################### 1) UNIFORM POLICY ####################
		# initialize V(s) to 0:
		V = {}
		for s in states:
			V[s] = 0

		# define the discount factor:
		gamma = 1.0 

		i = 0
		# repeat until convergence:
		while True:
			max_change = 0 # max change for the currenent iteration

			for s in states:
				# keep a copy of old V(s), s.t. we can keep track 
				# of the magnitude of each change:
				old_v = V[s]

				# NOTE: V(terminal_state) has no value:
				# check if not a terminal state:
				if s in grid.actions:
					# accumulate the value of this state:
					new_v = 0
					# we consider a UNIFORM policy,
					# i.e., the probability of taking any action is the same;
					p_a = 1.0 / len(grid.actions[s])
					# loop over all possible actions that can be taken
					# from the current state, s:

					for a in grid.actions[s]:
						# set our current state on the grid:
						grid.set_state(s)

						# make a move to get the reward, r, and next state, s':
						r = grid.move(a)
						s_prime = grid.current_state
						# for debugging:
						#print('s:', s, 's_prime:', s_prime, 'r:', r)

						# calculate (basically, accumulate) the Bellman equation:
						new_v += p_a * (r + gamma * V[s_prime])

					# update the value of the current state
					V[s] = new_v

					# update max_change:
					max_change = max(max_change, np.abs(old_v - V[s]))
			i += 1
			# check if converged:
			if max_change < THRESHOLD:
				break

		print('iterations to converge:', i, '\n')
		print('values for uniform policy:')
		print_values(V, grid)


	else:
		#################### 2) FIXED POLICY ####################
		# define our policy:
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

		# display the policy:
		print('the policy:')
		print_policy(policy, grid)
		print('\n')

		# initialize V(s) to 0:
		V = {}
		for s in states:
			V[s] = 0

		# define the discount factor:
		gamma = 0.9  # so now the further we get away from the winning state, 
					 # the smaller V(s) should be

		# repeat untill convergence:
		i = 0
		while True:
			max_change = 0 # maximum change for the current iteration
			# print('i:', i)
			for s in states:
				# copy the value of the current state 
				old_v = V[s]
				
				# NOTE: V(terminal_state) has no value:
				if s in policy:
					# set our state:
					grid.set_state(s)

					# take the action and receive a reward:
					a = policy[s]
					r = grid.move(a)
					s_prime = grid.current_state
					# for debugging:
					# print('s:', s, 's_prime:', s_prime, 'r:', r)

					# update the value of the state:
					V[s] = r + gamma * V[s_prime]

					# update the maximum change:
					max_change = max(max_change, np.abs(old_v - V[s]))
			i += 1
			# check if converged:
			if max_change < THRESHOLD:
				break

		print('iterations to converge:', i, '\n')
		print('values for fixed policy:')
		print_values(V, grid)



if __name__ == '__main__':
	policy = input('\nchoose policy (\'uniform\', \'fixed\'):\n')
	print('\n')

	main(policy=policy)
