import numpy as np 
import matplotlib.pyplot as plt 


class Grid:
	# the Environment
	def __init__(self, width, height, start):
		self.width = width
		self.height = height
		# the start position coordinates:
		self.i = start[0]
		self.j = start[1]

	def set_rewards_and_actions(self, rewards, actions):
		# reward should be a dict of (i, j): r  =  {(row, col): reward}
		# actions should be a dict of (i, j): A  =  {(row, col): list_of_possible_actions}
		self.rewards = rewards
		self.actions = actions # enumerate all the possible action that
							   # can take us to a new place

	def set_state(self, s):
		# needed for some RL algorithms,
		# e.g., the Iterative Policy Evaluation requires us
		# to get to the next state, do the action,
		# and then determine the next state - 
		# we need to figure state-transition probabilities by playing the game.
		self.i = s[0]
		self.j = s[1]

	@property	
	def current_state(self):
		# returns the current ij position of the Agent:
		return (self.i, self.j)

	def is_terminal(self, s):
		# returns a boolean:
		# if a given state is in the actions dict, 
		# return False - non-terminal,
		# b/c if you can do an action from the state,
		# then you can transition to a different state,
		# in which case the state is not terminal:
		return s not in self.actions

	def move(self, action):
		# first, check if the move is legal:
		if action in self.actions[(self.i, self.j)]:
			if action == 'U':
				self.i -= 1
			if action == 'D':
				self.i += 1
			if action == 'L':
				self.j -= 1
			if action == 'R':
				self.j += 1
		# return a reward if any:
		return self.rewards.get((self.i, self.j), 0)


	def undo_move(self, action):
		# these are the opposite of what the moves should normally do:
		# the user can pass the same action he/she has just inputted,
		# and the function will undo it:
		if action == 'U':
			self.i += 1
		if action == 'D':
			self.i -= 1
		if action == 'L':
			self.j += 1
		if action == 'R':
			self.j -= 1
		# in any undesirable case, raise an exception:
		# (the code should never reach here)
		assert(self.current_state in self.all_states)

	def game_over(self):
		# returns True if game is over (in a terminal state), else False
		return (self.i, self.j) not in self.actions 

	@property 
	def all_states(self):
		# get all states:
		# either a position that has possible next actions (non-terminal)
		# or a position that yields a reward
		return set(list(self.actions.keys()) + list(self.rewards.keys()))


def standard_grid():
	# define the "rules":
	# the reward for arriving at each state
	# and possible actions at each state.
	# 
	# the grid:
	#
	# . . . +1
	# . x . -1
	# s . . .
	#
	# x means a "wall" - you can't go there
	# s stands for the start position
	# +1 and -1 mean the reward at the corresponding state
	g = Grid(3, 4, (2, 0))
	rewards = {(0, 3): 1, (1, 3): -1}
	actions = {
		(0, 0): {'D', 'R'},
		(0, 1): {'R', 'L'},
		(0, 2): {'D', 'R', 'L'},
		(1, 0): {'U', 'D'},
		(1, 2): {'U', 'D', 'R'},
		(2, 0): {'U', 'R'},
		(2, 1): {'R', 'L'},
		(2, 2): {'U', 'R', 'L'},
		(2, 3): {'U', 'L'},
	}	
	g.set_rewards_and_actions(rewards, actions)
	return g


def negative_grid(step_cost=-0.1):
	# playing this game we want to 
	# try to minimize the number of moves - 
	# so we'll penalize each move:
	# (we've learned that we can make a robot
	#  solve the maze efficiently by penalizing
	#  every step taken).
	# 
	# NOTE: step_cost can't be positive, else the Agent 
	#       would be encouraged to do an infinite loop;
	#       step_cost also can't be less or equal to the 
	#       loosing state, else we'd lose asap.
	g = standard_grid()
	g.rewards.update({
		(0, 0): step_cost,
		(0, 1): step_cost,
		(0, 2): step_cost,
		(1, 0): step_cost,
		(1, 2): step_cost,
		(2, 0): step_cost,
		(2, 1): step_cost,
		(2, 2): step_cost,
		(2, 3): step_cost,
	})
	return g


def play_game(agent, env):
	pass


if __name__ == '__main__':
	g = standard_grid()







