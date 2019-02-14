import numpy as np 
import matplotlib.pyplot as plt 


LENGTH = 3


class Human:
	def __init__(self):
		pass 

	def set_symbol(self, symbol):
		self.symbol = symbol

	def take_action(self, env):
		# we need to take a legal move - place the symbol to a valid (empty) positon
		while True:
			pos = input('Enter coordinates i,j for your next move (i,j=0..2): ')
			i, j = pos.split(',')
			i = int(i)
			j = int(j)
			if env.is_empty(i, j):
				env.board[i,j] = self.symbol
				break
			else:
				print('Error: position occupied, enter another')
	
	def update(self, env):
		pass

	def update_state_history(self, s):
		pass



class Agent:
	def __init__(self, eps=0.1, alpha=0.5):
		# set epsilon - 
		# the probability of choosing a random action instead of greedy:
		self.eps = eps
		# learning rate:
		self.alpha = alpha
		self.verbose = False
		self.state_history = []

	def set_V(self, V):
		self.V = V

	def set_symbol(self, symbol):
		self.symbol = symbol

	def set_verbose(self, v):
		# v is a boolean:
		# if True, will print values per each position on the board
		self.verbose = v

	def reset_history(self):
		self.state_history = []

	def take_action(self, env):
		# first, check the board for valid positions
		# in order to later choose one:
		possible_moves = []
		for i in range(LENGTH):
			for j in range(LENGTH):
				if env.is_empty(i, j):
					possible_moves.append((i, j))

		# we're doing Epsilon Greedy:
		best_state = None
		if np.random.rand() < self.eps:
			# take a random action:
			if self.verbose:
				print('Taking a random action')			
			idx = np.random.choice(len(possible_moves))
			next_move = possible_moves[idx]
		else:
			# take an action with the maximum value:
			pos2value = {} # for drawing board with values if verbose
			next_move = None
			best_value = -1 # the value of the best state; 0 <= v <= 1
			for i, j in possible_moves:				
				# what is the state if we made this action (move)?
				env.board[i,j] = self.symbol 
				state = env.get_state()
				pos2value[(i,j)] = self.V[state]
				# SET IT BACK TO EMPTY!!!:
				env.board[i,j] = 0
				if self.V[state] > best_value:
					best_value = self.V[state]
					best_state = state
					next_move = (i, j)
			# for k, v in pos2value.items():
			# 	print(k, ':', v)
			if self.verbose:
				# draw the board with the values of available (empty) positions:
				print('Taking a greedy action')
				for i in range(LENGTH):
					print('---------------------------')
					for j in range(LENGTH):
						print(end=' ')
						if env.is_empty(i,j):
							# print the value
							print('%.2f  |' % pos2value[(i,j)], end=' ')
						else:
							if env.board[i,j] == env.x:
								print('x  |', end=' ')
							elif env.board[i,j] == env.o:
								print('o  |', end=' ')
							else:
								print('   |', end=' ')
					print()
				print('---------------------------')


		# take the action:
		env.board[next_move] = self.symbol

	def update_state_history(self, s):
		# cannot put this in take_action, b/c take_action only
		# happens once every other iteration for each player
		# but state history needs to be updated every iteration
		self.state_history.append(s)

	def update(self, env):
		'''Updates the value function.'''
		# we want to backtrack over the states:
		# V(prev_state) += alpha*(V[next_state] - V[prev_state]),
		# where V(next_state) = reward, if it's the most current state
		
		# NOTE: we do this only at the END of an episode,
		# not so for all the algorithms we'll study
		reward = env.reward(self.symbol) # the most current state's value
		target = reward
		for prev in reversed(self.state_history):
			value = self.V[prev] + self.alpha*(target - self.V[prev])
			self.V[prev] = value
			target = value
		self.reset_history()



class Environment:	
	def __init__(self):
		self.winner = None # there is no winner yet
		self.board = np.zeros((LENGTH, LENGTH)) # 0 is to be our symbol for 'empty'
		self.num_states = 3**self.board.size
		self.ended = False # will become True if game is over
		# define the symbols for x and o:
		# should be numbers, since we're to store them in a np.array
		self.x = -1 # represents an x on the board, player 1
		self.o = 1 # represents an o on the board, player 2

	def is_empty(self, i, j):
		# before trying to put down its own pieces, 
		# an Agent must check whether the position is empty
		return self.board[i,j] == 0

	def reward(self, symbol):
		# no reward until game (an episode) is over:
		if not self.game_over():
			return 0

		# if we are here, game is over
		# symbol will be self.x or self.o
		return 1 if self.winner == symbol else 0

	def get_state(self):
		'''Returns a state - a single unique integer (a hashed int).
		We have 3 possible states per position - 0, 1, or 2.
		See 'representing_states.py' for more details.'''
		N = len(self.board)
		# print(N)
		h = 0 # hash
		k = 0
		for i in range(N):
			for j in range(N):
				# decode the content of a position on the board
				if self.board[i,j] == 0:
					# if the position is empty:
					v = 0
				elif self.board[i,j] == self.x:
					v = 1
				elif self.board[i,j] == self.o:
					v = 2
				h += (3**k)*v
				k += 1
		return h
	
	def game_over(self, force_recalculate=False):
		'''Checks if there is a winner or it's a draw.
		Returns a boolean.'''
		if not force_recalculate and self.ended:
			return self.ended
					
		# check the horizontal lines:
		for i in range(LENGTH):
			for p in (self.x, self.o): # for each player				
				if self.board[i,:].sum() == p*LENGTH:
					self.winner = p
					self.ended = True
					return True					
				
		# check the vertical lines:
		for j in range(LENGTH):
			for p in (self.x, self.o): # for each player				
				if self.board[:,j].sum() == p*LENGTH:
					self.winner = p
					self.ended = True
					return True
			
		# check the diagonals:			
		for p in (self.x, self.o): # for each player
			# 1) main diagonal:
			if self.board.trace() == p*LENGTH:
				self.winner = p
				self.ended = True
				return True
			# 2) the other one:
			elif np.rot90(self.board).trace() == p*LENGTH:
				# we could also use np.flip(self.board, axis=1).trace()
				self.winner = p
				self.ended = True
				return True
		
		# check if draw:
		# NOTE: draw if there is no empty positions AND 
		# 		if is not the initial state
		if np.all((self.board == 0) == False):
			self.winner = None
			self.ended = True
			return True
		
		# if game is not over:
		self.winner = None
		return False

	def draw_board(self):
		for i in range(LENGTH):
			print('----------')
			for j in range(LENGTH):
				print(end=' ')
				if self.board[i,j] == self.x:
					print('x', end=' ')
				elif self.board[i,j] == self.o:
					print('o', end=' ')
				else:
					print(' ', end=' ')
			print()
		print('----------')



def get_state_hash_and_winner(env, i=0, j=0):
	'''Inputs: env is needed just for the board matrix;
	i and j are coordinates of a position on the board,
	needed to place a new value (0, env.o or env.x).
	Returns tripleS (state, winner, ended)'''
	results = []

	for v in (0, env.x, env.o): # by analogy with "prefix in ('0', '1')"
		env.board[i, j] = v # should be 0 if the board is empty
		if j == 2:
			# j sets back to 0, i increases untill i = 2, then we're done:
			if i == 2:
				# the Base Case: the board is full;
				# collect and return the results:
				state = env.get_state() # a hashed int
				ended = env.game_over(force_recalculate=True) # a boolean
				winner = env.winner # p1 - env.x, p2 - env.o, or None
				results.append((state, winner, ended))
			else:
				# set j = 0, increment i by 1
				results += get_state_hash_and_winner(env, i + 1, 0)
		else:
			# increment j, i stays the same
			results += get_state_hash_and_winner(env, i, j + 1)

	return results


def initV_x(env, state_winner_triples):
	# initialize state values as follows:
	# V(s) = 1, if x (p1) wins
	# V(s) = 0, if x (p1) loses or draw
	# V(s) = 0.5, otherwise
	V = np.zeros(env.num_states)
	for state, winner, ended in state_winner_triples:
		if ended:
			if winner == env.x:
				v = 1
			else:
				v = 0
		else:
			v = 0.5
		V[state] = v
	return V


def initV_o(env, state_winner_triples):
	# initialize state values as follows:
	# V(s) = 1, if o (p2) wins
	# V(s) = 0, if o (p2) loses or draw
	# V(s) = 0.5, otherwise
	V = np.zeros(env.num_states)
	for state, winner, ended in state_winner_triples:
		if ended:
			if winner == env.o:
				v = 1
			else:
				v = 0
		else:
			v = 0.5
		V[state] = v
	return V


def play_game(p1, p2, env, draw=False):
	''' Loops until the game is over.'''
	current_player = None
	while not env.game_over():
		# alternate between two players;
		# p1 always starts first:
		if current_player == p1:
			current_player = p2
		else:
			current_player = p1

		# draw the board before the user wants to see it makes a move:
		if draw:
			if draw == 1 and current_player == p1:
				env.draw_board()
			if draw == 2 and current_player == p2:
				env.draw_board()
	
		# current player makes a move:
		current_player.take_action(env)

		# update state histories:
		state = env.get_state()
		p1.update_state_history(state)
		p2.update_state_history(state)

	if draw:
		env.draw_board()

	# do the value function update:
	p1.update(env)
	p2.update(env)


if __name__ == '__main__':
	p1 = Agent(eps=0.1)
	p2 = Agent()	

	env = Environment()
	state_winner_triples = get_state_hash_and_winner(env)
	# print('state_winner_triples:', state_winner_triples[:10])
	# exit()
	# initialize value function for each Agent:
	Vx = initV_x(env, state_winner_triples)
	# print('Vx:', Vx[:10])
	p1.set_V(Vx)
	Vo = initV_o(env, state_winner_triples)
	# print('Vo:', Vo[:10])
	p2.set_V(Vo)
	# exit()

	# give each player a corresponding symbol:
	p1.set_symbol(env.x)
	p2.set_symbol(env.o)

	# Ngames = 10000
	# print("Training the AI for {} games".format(Ngames))
	# for i in range(Ngames):		
	# 	play_game(p1, p2, Environment())
	# 	if i % 500 == 0:
	# 		print(i)


	print('\nPlay with the AI')
	# play with the trained AI:	
	h = Human()
	h.set_symbol(env.o)
	p1.set_verbose(True)

	while True:
		play_game(p1, h, Environment(), draw=2)
		prompt = input('Play again? [Y/n]: ')
		if prompt in ['N', 'n']:
			break

