import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
#import theano.tensor as T

from theano_ann import ANN  
from sklearn.utils import shuffle
from datetime import datetime 


def random_search():  
    # get the data:
    df = pd.read_csv('Spirals_dataset.csv')
    data = df.as_matrix()
    X = data[:,:-1]
    Y = data[:,-1]

    # visualize the data:
    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.title('Spirals')
    plt.axis('equal')
    plt.show()

    # split the data:
    Xtrain, Ytrain = X[:-270, :], Y[:-270]
    Xtest, Ytest = X[-270:, :], Y[-270:]

    # initial hyperparameters:
    M = 20  # number of hidden layer neurons
    num_hl = 2
    log_lr = -4  # learning rate
    log_l2 = -2 # L2-regularization term
    max_tries = 1000
    

    best_validation_score = 0
    best_hl_size = None
    best_lr = None
    best_l2 = None
    t0 = datetime.now()
    # Random Search loop:
    for _ in range(max_tries):
        model = ANN([M]*num_hl)
        model.fit(
            Xtrain,
            Ytrain, 
            learning_rate=10**log_lr, 
            reg=10**log_l2, 
            mu=0.9, 
            epochs=3000, 
            show_fig=False
        )
        validation_score = model.score(Xtest, Ytest)
        train_score = model.score(Xtrain, Ytrain)
        print('\nvalidation set accuracy: %.3f,  training set accuracy: %.3f' % (validation_score, train_score))   
        print('hidden layer size: {}, learning rate: {}, l2: {}'.format([M]*num_hl, 10**log_lr, 10**log_l2))

        if validation_score > best_validation_score:
            best_validation_score = validation_score
            best_hl_size = M
            best_num_hl = num_hl
            best_log_lr = log_lr
            best_log_l2 = log_l2

        # update the hyperparameters:
        log_lr = best_log_lr + np.random.randint(-1, 2)
        log_l2 = best_log_l2 + np.random.randint(-1, 2)
        M = best_hl_size + np.random.randint(-1, 2)*10 # -10, 0, or 10
        M = max(10, M) # in case if M has been updated to 0
        num_hl = best_num_hl + np.random.randint(-1, 2)
        num_hl = max(1, num_hl) # in case if num_hl has been updated to 0


    dt = datetime.now() - t0
    print('\nElapsed time:', dt)
    print('\nBest validation accuracy:', best_validation_score)
    print('\nBest settings:')
    print('hidden layer size:', best_hl_size)
    print('number of hidden layers:', best_num_hl)
    print('learning rate:', 10**best_log_lr)
    print('l2:', 10**best_log_l2)
    print()



if __name__ == '__main__':
    random_search()