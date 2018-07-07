import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat

mnist_data = '/mnist_train.csv'
fer_data = '/fer2013.csv'

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 / 2.0)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def init_filter_(shape, mode='theano'):
    if mode == 'theano':
        # filters dimensionality:
        # num_filters, num_color_channels, filter_width, filter_height
        w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:])/2.0)
    if mode == 'tensorflow':
        # filters dimensionality:
        #  filter_width, filter_height, num_color_channels, num_filters
        w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1])/2.0)
    return w.astype(np.float32)


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)


def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def cost(T, Y):
    return -(T*np.log(Y)).sum()


def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()


def error_rate(targets, predictions):
    return np.mean(targets != predictions)


def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i].astype(np.int32)] = 1
    return ind


def crossValidation(model, X, Y, K=5):
    # split data into K parts
    X, Y = shuffle(X, Y)
    sz = len(Y) // K
    errors = []
    for k in range(K):
        xtr = np.concatenate([ X[:k*sz, :], X[(k*sz + sz):, :] ])
        ytr = np.concatenate([ Y[:k*sz], Y[(k*sz + sz):] ])
        xte = X[k*sz:(k*sz + sz), :]
        yte = Y[k*sz:(k*sz + sz)]

        model.fit(xtr, ytr)
        err = model.score(xte, yte)
        errors.append(err)
    print("errors:", errors)
    return np.mean(errors)


def swish(a, tensorflow=False):
    if tensorflow:
        import tensorflow as tf
        return a * tf.sigmoid(a)
    else:
        return a / (1 + np.exp(-a))


def get_normalized_data(shape='theano', show_img=False):
    print('Reading in and transforming data...')
    df = pd.read_csv(mnist_data)
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]
    N, D = X.shape
    if show_img:
        n = np.random.randint(N)
        plt.imshow(X[n,:].reshape(28,28), cmap='gray')
        plt.title("Original image of '{}'".format(int(Y[n])))
        plt.show()
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std==0, 1) # replace entries of std=0 to 1
    X = (X - mu) / std
    if show_img:        
        plt.imshow(X[n,:].reshape(28,28), cmap='gray')
        plt.title("Normalized image of '{}'".format(int(Y[n])))
        plt.show()
    if shape == 'theano':
        X = X.reshape(N, 1, 28, 28)

    return X, Y


def getData(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    for line in open(fer_data):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y


def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y
    