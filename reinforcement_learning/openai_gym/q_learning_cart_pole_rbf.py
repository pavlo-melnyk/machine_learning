import os 
import sys
import gym 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 

from gym import wrappers
from mpl_toolkits.mplot3d import Axes3D 
# from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.utils import shuffle
from datetime import datetime 



class SGDRegressor:
    def __init__(self, learning_rate='constant', loss='squared_loss', penalty='none', 
                 reg=0.0001, shuffle=True, verbose=False, random_state=None, 
                 lr0=0.01, n_iter=None, display_cost=False):
                 
        self.w = np.empty(0) # the model parameters
        self.learning_rate = learning_rate
        self.lr0 = lr0
        self.loss = loss 
        self.penalty = penalty
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
        self.reg = reg
        self.n_iter = n_iter
        self.display_cost = display_cost

        if self.loss == 'squared_loss':
            self.loss = self.squared_loss

        if self.learning_rate == 'constant':
            self.lr = self.lr0

        if self.penalty in ['none', None]:
            self.reg = 0


    def squared_loss(self, t, y):
        ''' Regularized Squared Loss. '''
        delta = t - y
        return np.sum(delta.dot(delta)) + self.reg*self.w.dot(self.w)


    def fit(self, X, t):
        N, D = X.shape
        # add a bias column of ones to the input data:
        X = np.concatenate((np.ones((N, 1)), X), axis=1)
        D += 1

        assert len(t.shape) == 1

        if self.shuffle:
            X, t = shuffle(X, t)

        if len(self.w) == 0:
            # initialize the model parameters:
            self.w = np.random.randn(D)/np.sqrt(D)

        ll = []
        # Gradient Descent:       
        for i in range(self.n_iter): 
            self.w -= self.lr*( X.T.dot(self.predict(X) - t) +  self.reg*self.w )
            # calculate the loss:
            l = self.loss(t, self.predict(X))
            ll.append(l)
            if self.verbose:
                print('iter: %d,  cost: %.6f,  acc: %.3f' % (i, l, self.score(X, t)))

        if self.display_cost:
            plt.plot(ll, label=self.loss)
            plt.xlabel('iterations')
            plt.legend()
            plt.show()
    

    def partial_fit(self, x, t):
        assert len(x.shape) == 2
        D = x.shape[1]
        # add a 1 for bias to the input data:
        x = np.concatenate((np.ones((1, 1)), x), axis=1)
        D += 1

        if len(self.w) == 0:
            # print('initializing the model parameters')
            self.w = np.random.randn(D)/np.sqrt(D)
            # print(self.w)

        # make a GD update:
        self.w -= self.lr*( (self.predict(x) - t).dot(x) +  self.reg*self.w )


    def predict(self, X):
        N, D = X.shape
        try:
            return X.dot(self.w)
        except Exception:
            try:
                # in case called not from the either fit function,
                # add a bias column to the input data:
                X = np.concatenate((np.ones((N, 1)), X), axis=1)
                return X.dot(self.w)
            except Exception:
                # in case the weights are not initialized:
                assert len(self.w) == 0
                print('Parameters are not initialized!\nYou must first fit some data to the model')

    
    def score(self, X, t):
        return np.mean(self.predict(X) == t)


    
class FeatureTransformer:
    def __init__(self, env):
        # guess a plausible range of states:
        state_samples = np.random.random((20000, 4))*2 - 2
        
        # scale the collected data, s.t. feature_mean = 0, feature_var = 1:
        scaler = StandardScaler()
        scaler.fit(state_samples)
        # print('scaler.mean_:', scaler.mean_)
        scaled_samples = scaler.transform(state_samples)
        # print('scaled_samples.mean(axis=0):', scaled_samples.mean(axis=0))
                
        # a single RBF kernel is 
        # 
        #            f(x) = exp( -gamma * ||x-c||^2 ) ,
        # 
        # where c     - is a center/exemplar/component that lives in
        #               the same space as x;
        #       gamma - a shape parameter.

        # NOTE: RBFSampler from SciKit-Learn just APPROXIMATES the RBF
        #       and doesn't really use training data (input to the fit function);
        #       check the following link for details: 
        #       https://www.kaggle.com/sy2002/rbfsampler-actually-is-not-using-any-rbfs

        # RBF kernels with different variances are to cover different states' data space:
        transformer = FeatureUnion([
            ('rbf_1', RBFSampler(gamma=0.05, n_components=1000)),
            ('rbf_2', RBFSampler(gamma=0.1, n_components=1000)),
            ('rbf_3', RBFSampler(gamma=0.5, n_components=1000)),
            ('rbf_4', RBFSampler(gamma=1.0, n_components=1000)),
        ])

        # in order to investigate the dimensionality,
        # transform the scaled samples collected before:
        example_features = transformer.fit_transform(scaled_samples)
        self.dimensionality = example_features.shape[1]
        self.scaler = scaler
        self.transformer = transformer
        
    def transform(self, states):
        ''' States must be a 2D array. '''
        scaled_data = self.scaler.transform(states)
        return self.transformer.transform(scaled_data)



class Model:
    def __init__(self, env, feature_transformer, lr, lr0):
        self.env = env 
        self.feature_transformer = feature_transformer        
        # we'll have a collection of models - 
        # one model for each action!
        self.rbf_models = []
        for i in range(env.action_space.n):
            rbf_model = SGDRegressor(learning_rate=lr, lr0=lr0)
            # initialize the weights of the linear model - fit some data:
            x = feature_transformer.transform( [env.reset()] )
            rbf_model.partial_fit(x, [0])
            self.rbf_models.append(rbf_model)
            
    def update(self, s, a, G):
        ''' Performs a Q-Learning semigradient update.'''
        # transform the state into a feature vector:
        x = self.feature_transformer.transform([s]) # 2D-vector
        # make a gradient descent update for the given model:
        self.rbf_models[a].partial_fit(x, [G]) # targets should be passed as 1D
                                               # so we ‘wrap’ the scalar G in a list
        
    def predict(self, s):
        # transform the state into a feature vector:
        x = self.feature_transformer.transform([s]) # data input must be 2D
        return np.array([model.predict(x)[0] for model in self.rbf_models])

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
            r = -100

        # update the parameters of the model:
        G = r + gamma*np.max(model.predict(s_prime)) # the estimated value of the return
        model.update(s, a, G)
        
        # the next state becomes current:
        s = s_prime
        
        steps += 1

    return steps, total_reward
        


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean( total_rewards[max(0, t-100): t+int(t==0)] )
    plt.plot(running_avg)
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

    model = Model(env, feature_transformer, 'constant', 0.1)

    gamma = 0.9 # the discount factor
    n_episodes = 500
    avg_length = 0
    total_rewards = np.empty(n_episodes)
    
    for t in range(n_episodes):
        t0 = datetime.now()
        # eps = 0.1*(0.97**t)
        eps = 1/np.sqrt(t+1)
        steps, total_reward = play_game(model, eps, gamma)

        total_rewards[t] = total_reward
        avg_length += (steps - avg_length)/(t+1)
        if t % 50 == 0:
            print('episode: %d\tETA: %s\ttotal reward: %.3f\teps: %.3f' % \
                (t, datetime.now() - t0, total_reward, eps))

    print('\navg episode length:', avg_length)
    print('\navg reward for last 100 episodes:', np.mean(total_rewards[-100:]))
    
    plt.plot(total_rewards)
    plt.title('Rewards')
    plt.xlabel('episodes')
    plt.show()

    plot_running_avg(total_rewards)



if __name__ == '__main__':
    main()