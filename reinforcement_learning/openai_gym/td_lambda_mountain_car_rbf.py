import os 
import sys
import gym 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 

from gym import wrappers
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from datetime import datetime 



class FeatureTransformer:
    def __init__(self, env):        
        # gather 10^4 samples from the state-space:
        state_samples = np.array([env.observation_space.sample() for i in range(10000)])
        # scale the collected data, s.t. mean = 0, var = 1:
        scaler = StandardScaler()
        scaler.fit(state_samples)
        scaled_samples = scaler.transform(state_samples)

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

        # RBF kernels with different variances are to cover different state's data space:
        transformer = FeatureUnion([
            ('rbf_1', RBFSampler(gamma=0.5, n_components=500)),
            ('rbf_2', RBFSampler(gamma=1.0, n_components=500)),
            ('rbf_3', RBFSampler(gamma=2.0, n_components=500)),
            ('rbf_4', RBFSampler(gamma=5.0, n_components=500)),
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



class ETRegressor:
    ''' Sort of SGDRegressor but with Eligibility Trace.
    Eligibility Trace is a vector the same size as 
    parameter vector. It keeps track of old gradient,
    similar to momentum in deep learning:
                e(0) = 0; 
    e(t) = grad(V(s(t)), param) + gamma*lambda*e(t-1) .

    lambda tells us the portion of the "past" we want
    to keep.
    '''
    def __init__(self, D, learning_rate, gamma, lmb):
        self.lr = learning_rate
        self.gamma = gamma
        self.lmb = lmb
        # initialize the parameters:
        self.w = np.random.randn(D, 1)/np.sqrt(D)
        # print('self.w.shape:', self.w.shape)
        # exit()
        # eligibility vector:
        self.elig = np.zeros_like(self.w)


    def partial_fit(self, x, t):        
        # parameter update looks like:
        # e(t) = grad(V(s(t)), theta(t-1)) + gamma*lambda*e(t-1)
        # theta(t) = theta(t-1) + lr*e(t)*[t - predict(x)]
        self.elig = x.T + self.gamma*self.lmb*self.elig        
        self.w += self.lr*self.elig*(t - self.predict(x))
        # print('self.w.shape:', self.w.shape)
        # exit()

    def predict(self, X):
        return X.dot(self.w)



class Model:
    def __init__(self, env, feature_transformer, lr, gamma, lmb):
        self.env = env 
        self.feature_transformer = feature_transformer        
        # we'll have a collection of models - 
        # one model for each action!
        self.models = []
        for i in range(env.action_space.n):
            model = ETRegressor(feature_transformer.dimensionality, lr, gamma, lmb)
            # # fit some data:
            # x = feature_transformer.transform( [env.reset()] )
            # model.partial_fit(x, [0])
            self.models.append(model)
            

    def update(self, s, a, G):
        ''' Performs a TD(lambda) update.'''
        # transform the state into a feature vector:
        x = self.feature_transformer.transform([s]) # 2D-vector
        # make a gradient descent update for the given model:
        self.models[a].partial_fit(x, [G]) # targets should be passed as 1D
                                           # so we ‘wrap’ the scalar G in a list
        

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
    steps = 0
    done = False
    while not done:
        # choose an action:        
        a = model.epsilon_greedy_action(s, eps)

        # take the action:
        s_prime, r, done, _ = model.env.step(a)
               
        # update the parameters of the model:
        G = r + gamma*np.max(model.predict(s_prime)) # the estimated value of the return
        model.update(s, a, G)

        # the next state becomes current:
        s = s_prime

        total_reward += r
        steps += 1

    return steps, total_reward
        


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean( total_rewards[max(0, t-99):t+int(t==0 or t>=100)] )
    plt.plot(running_avg)
    plt.title('Running Average')
    plt.show()



def plot_cost_to_go(model, num_tiles=20):
    ''' Plots negative of the Optimal Value Function V*(s). '''
    x = np.linspace(model.env.observation_space.low[0], model.env.observation_space.high[0], num_tiles)
    y = np.linspace(model.env.observation_space.low[1], model.env.observation_space.high[1], num_tiles)
    X, Y = np.meshgrid(x, y) # each of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda v: -np.max(model.predict(v)), 2, np.dstack([X, Y]))
    # print('Z.shape:', Z.shape) # (num_tiles, num_tiles)
    # exit()

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, 
        rstride=1, cstride=1, cmap='coolwarm', edgecolors='k')#, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('-V*(s)')
    ax.set_title('Cost-To-Go Function')
    fig.colorbar(surf)
    plt.show()



def main():
    env = gym.make('MountainCar-v0')
    # set max number of steps per episode:
    env._max_episode_steps = 500
	
    feature_transformer = FeatureTransformer(env)
    # print('feature.dimensionality:', feature_transformer.dimensionality)
    # s = env.reset()
    # feature = feature_transformer.transform([s])
    # print('feature.shape:', feature.shape)
    
    # if required, save the video of our Agent playing the game:
    filename = os.path.basename(__file__).split('.')[0]
    save_as = filename + ' ' + str(datetime.now()).replace(':', ' ') # enable for windows
    if 'wrap' in sys.argv:
        env = wrappers.Monitor(env, save_as)

    # define some hyperparameters and instantiate our TD(lambda) model:
    gamma = 0.99 # the discount factor
    lmb = 0.7 # the lambda parameter of TD(lambda)
    model = Model(env, feature_transformer, 0.01, gamma, lmb)
    
    n_episodes = 500
    avg_length = 0
    total_rewards = np.empty(n_episodes)
    
    for t in range(n_episodes):
        t0 = datetime.now()
        eps = 0.1*(0.97**t)
        # eps = 0 # using optimistic initial values method
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

    plot_cost_to_go(model)



if __name__ == '__main__':
    main()