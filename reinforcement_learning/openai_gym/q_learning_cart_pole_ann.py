import os 
import sys
import gym 
import tensorflow as tf
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 

from gym import wrappers
from tf_ann import ANN
from sklearn.preprocessing import StandardScaler
from datetime import datetime



class FeatureTransformer:
    def __init__(self, env):
        # guess a plausible range of states:
        state_samples = np.random.random((20000, 4))*2 - 2

        # scale the collected data, s.t. feature_mean = 0, feature_var = 1:
        scaler = StandardScaler()
        scaler.fit(state_samples)
        scaled_samples = scaler.transform(state_samples)
        # print('scaled_samples.mean(axis=0):', scaled_samples.mean(axis=0))
        
        self.dimensionality = scaled_samples.shape[1]
        self.scaler = scaler


    def transform(self, states):
        ''' States must be a 2D array. '''
        return self.scaler.transform(states)



class Model:
    def __init__(self, env, feature_transformer, hidden_layer_sizes, dropout_rates, activation, lr, l2):
        self.env = env
        self.feature_transformer = feature_transformer
        self.hidden_layer_sizes = hidden_layer_sizes
        # we'll have a collection of models - 
        # one model for each action!
        self.models = []
        for i in range(env.action_space.n):
            model = ANN(feature_transformer.dimensionality, hidden_layer_sizes,
                        dropout_rates, activation, lr, l2)
            self.models.append(model)
            

    def update(self, s, a, G):
        ''' Performs a Q-Learning semigradient update.'''
        # transform the state into a feature vector:
        x = self.feature_transformer.transform([s]) # 2D-vector
        # make a gradient descent update for the given model:
        cost = self.models[a].partial_fit(x, [G]) # targets should be passed as 1D
                                                  # so we ‘wrap’ the scalar G in a list
        return cost


    def predict(self, s):
        # transform the state into a feature vector:
        x = self.feature_transformer.transform([s]) # data input must be 2D
        return np.array([model.predict(x)[0] for model in self.models])


    def epsilon_greedy_action(self, s, eps):
        p = np.random.random()
        if p < eps:
            return self.env.action_space.sample()

        else:
            # choose the action (model) that yields the biggest value:
            actions = self.predict(s)
            return np.argmax(actions)



def play_game(model, eps, gamma):
    total_reward = 0
    s = model.env.reset()
    ll = [] # cost storage    
    steps = 0
    done = False
    while not done:
        # choose an action:        
        a = model.epsilon_greedy_action(s, eps)

        # take the action:
        s_prime, r, done, _ = model.env.step(a)
        
        total_reward += r       
        
        if done and steps < 499:
            # penalize our Agent for not reaching the maximum # of steps:
            r = -400

        # update the parameters of the model:
        G = r + gamma*np.max(model.predict(s_prime)) # the estimated value of the return
        loss = model.update(s, a, G)
        ll.append(loss)

        # the next state becomes current:
        s = s_prime
        
        steps += 1

    return ll, steps, total_reward
        


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean( total_rewards[max(0, t-99):t+int(t==0 or t>=100)] )
    plt.plot(running_avg)
    plt.xlabel('episodes')
    plt.title('Running Average')
    plt.show()



def main():
    env = gym.make('CartPole-v1')

    # max number of steps per episode:
	# print('\ninitial steps_limit:', env._max_episode_steps)

    # check the state space borders:
    # for i in range(4):
    #     print(env.observation_space.low[i], env.observation_space.high[i])
    # exit()

    feature_transformer = FeatureTransformer(env)
    
    # if required, save the video of our Agent playing the game:
    filename = os.path.basename(__file__).split('.')[0]
    save_as = filename + ' ' + str(datetime.now()).replace(':', ' ') # enable for windows
    if 'wrap' in sys.argv:
        env = wrappers.Monitor(env, save_as)
    
    # create our model:
    hidden_layer_sizes = [4000]
    dropout_rates = None 
    # dropout_rates = [1, 0.8, 0.8, 0.8, 0.8, 0.8]
    activation = tf.nn.tanh # for hidden layers
    # activation = tf.nn.relu 
    lr = 1e-5
    l2 = 0 # l2-regularization scaler
    model = Model(env, feature_transformer, hidden_layer_sizes, dropout_rates, activation, lr, l2)

    gamma = 0.99 # the discount factor
    n_episodes = 20000
    avg_length = 0
    total_rewards = np.empty(n_episodes)
    LL = []
    T0 = datetime.now()
    for t in range(n_episodes):
        t0 = datetime.now()
        # eps = 0.1*(0.97**t)
        eps = 1/np.sqrt(t+1)
        ll, steps, total_reward = play_game(model, eps, gamma)
        LL += ll
        total_rewards[t] = total_reward
        avg_length += (steps - avg_length)/(t+1)
        if t % 100 == 0:
            print('episode: %d\tETA: %s\ttotal reward: %.3f\teps: %.3f' % \
                (t, datetime.now() - t0, total_reward, eps))
            print('avg reward over last 100:', np.mean( total_rewards[max(0, t-99):t+int(t==0 or t>=100)] ))
    
    # print('\navg episode length:', avg_length)
    print('\navg reward for last 100 episodes:', np.mean(total_rewards[-100:]))
    print('\nETA:', datetime.now() - T0)
    
    plt.plot(total_rewards)
    plt.title('Rewards')
    plt.xlabel('episodes')
    plt.show()

    plot_running_avg(total_rewards)

    # plot the cost:
    # plt.plot(LL)
    # plt.xlabel('steps')
    # plt.title('Squared Loss')
    # plt.show()



if __name__ == '__main__':
    main()
