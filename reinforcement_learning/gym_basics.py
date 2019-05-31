import gym
import numpy as np 


# 1) get our environment:
env = gym.make('CartPole-v1')

# the full list of envs here:
# https://gym.openai.com/envs


# 2) start an episode:

# put the Agent in the start state
s = env.reset() # also returns the state
print('\nthe start state:', s)

# what do these numbers mean?
# https://github.com/openai/gym/wiki/CartPole-v0

# Num	Observation	 	       Min	       Max
# 
# 0	    Cart Position	       -2.4	       2.4
# 1	    Cart Velocity	       -Inf	       Inf
# 2	    Pole Angle	           ~ -41.8°	   ~ 41.8°
# 3	    Pole Velocity At Tip   -Inf	       Inf

# we can get some useful info from the console as well:
box = env.observation_space
# print('\n', box)
# box. # type in the console and hit [Tab]

# similar with the actions:
actions = env.action_space


# 3) play an episode:

# observation, reward, done, info = env.step(action)

# the info dictionary is typically ignored

# let's play the game taking random actions;
# this will end quickly, b/c random actions can't keep 
# the pole up for long:
avg_steps = 0
print('\nplaying the game taking random actions...')
for t in range(10000):
	i = 0 # number of steps taken in an episode
	# play an episode;
	done = False
	while not done:
		observation, reward, done, _ = env.step(actions.sample())
		i += 1
	# reset the environment once the game is over:
	env.reset()
	avg_steps += (i - avg_steps)/(t+1)

print('\navg num of steps:', int(avg_steps))




