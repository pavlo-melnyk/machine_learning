''' 
The script is to show the difference between the outputs 
of the RBFSampler from SciKit-Learn and a manually programmed RBF kernel.
Inspired by and following the guideline of
https://www.kaggle.com/sy2002/rbfsampler-actually-is-not-using-any-rbfs
'''

import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.kernel_approximation import RBFSampler



def rbf(x, c, gamma):
    ''' 
    Takes in a data vector x, centroid/exemplar c, 
    and the precision parameter gamma = 1/variance.
    Computes and returnes a Gaussian RBF function 
            f(x) = exp( -gamma * ||x-c||^2 ) 
    - the exp of the scaled negative squared distance
    between x and c.
    '''
    x, c = np.array(x), np.array(c)
    dist = np.linalg.norm(x - c) # a float scalar value
    return np.exp( -gamma * dist**2 )



def main():
    # use num_x * num_y 2D samples:
    num_x, num_y = 200, 200
   
    # the samples will consist of values
    # running from min_val to max_val:
    min_val, max_val = -2, 2

    # generate a mesh grid of sample values:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.mgrid.html
    # "However, if the step length is a complex number (e.g. 5j), 
    # then the integer part of its magnitude is interpreted
    # as specifying the number of points to create between
    # the start and stop values, where the stop value is inclusive."
    # NOTE: plt.imshow() draws an image of the format [rows, columns, depth],
    #       from top-left to bottom-right, i.e., 
    #       from (y_max_val, x_min_val) to (y_min_val, x_max_val)
    #       https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.imshow.html
    grid_y, grid_x = np.mgrid[
        max_val:min_val:complex(num_y),
        min_val:max_val:complex(num_x)
        ]

    grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
    # print('grid_x.shape, grid_y.shape:', grid_x.shape, ',', grid_y.shape)
    # print(grid_x[:10], '\n', grid_y[:10])
    


    ########################## Use Manually Programmed RBF ##########################
    # apply the RBF to the generated samples:
    img = [] # for visualization
    for x, y in zip(grid_x, grid_y):
        sample = [x, y]
        img.append( rbf(sample, c=[0.75, 0.75], gamma=1.0) )
    
    img = np.reshape(img, (num_x, num_y))

    # visualize the result:
    # plt.imshow(img, cmap='gray', extent=2*[min_val, max_val], interpolation='none')
    # plt.show()



    ########################## Use SciKit-Learn RBFSampler ##########################
    # trying to repeat the set-up of the previous step:
    rbf_sampler = RBFSampler(gamma=1.0, n_components=1) # n_components = # of centers
    
    # the fit function uses only the dimensionality of the input data,
    # which should be a 2D array:
    rbf_sampler.fit(np.random.random((1, 2)))   
    # print(rbf_sampler.transform([[0, 0]]).shape) 

    # apply the RBF to the same samples as before:
    img_2 = []
    for x, y in zip(grid_x, grid_y):
        sample = [x, y]
        output = rbf_sampler.transform([sample]) # shape=(1,1)
        img_2.append( output.ravel()[0] )

    img_2 = np.reshape(img_2, (num_x, num_y))

    # unlike the manually programmed RBF, SciKit-Learn implementation
    # can output non-normalized negative values;
    # let's normalize it:
    img_2 -= img_2.min() # now all values are +ve
    img_2 /= img_2.max() # now all values are in [0, 1]

    # visualize the result:
    # plt.imshow(img_2, cmap='gray', extent=2*[min_val, max_val], interpolation='none')
    # plt.show()    

    # visualize the results of the two methods together:
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', extent=2*[min_val, max_val], interpolation='none')
    plt.title('Manually Programmed RBF')
    plt.subplot(1, 2, 2)
    plt.title('SciKit-Learn RBFSampler')
    plt.imshow(img_2, cmap='gray', extent=2*[min_val, max_val], interpolation='none')
    plt.show()



    ################### Create a network of Manually Programmed RBFs ###################
    N_COMPONENTS = 100 # # of centers/exemplars
    GAMMA = 5.0

    # create N_COMPONENTS random centers for RBFs:
    C_width = max_val - min_val
    C = [[np.random.random()*C_width - C_width/2.0,
          np.random.random()*C_width - C_width/2.0] for i in range(N_COMPONENTS)]
    # print('C:', C)
    # exit()

    # calculate the RBFs and visualize by summing up all distances to all centers
    # per sample:
    img_3 = []
    for x, y in zip(grid_x, grid_y):
        sample = [x, y]
        img_3.append(np.sum( [rbf(sample, c, GAMMA) for c in C] ))

    # reshape and normalize to [0, 1]:
    img_3 = np.reshape(img_3, (num_x, num_y))
    img_3 /= img_3.max()

    # visualize:
    # plt.imshow(img_3, cmap='gray', extent=2*[min_val, max_val], interpolation='none')
    # plt.title('Outputs of %d Manually Programmed RBFs with Random Exemplars' % N_COMPONENTS)
    # plt.show()



    ################### Create a network of SciKit-Learn RBFSamples ###################
    # trying to repeat the set-up of the previous step:
    rbf_samplers = RBFSampler(gamma=GAMMA, n_components=N_COMPONENTS)
    rbf_samplers.fit(np.random.random((1, 2)))

    # repeat the visualization process:
    img_4 = []
    for x, y in zip(grid_x, grid_y):
        sample = [x, y]
        output = rbf_samplers.transform([sample])
        img_4.append(np.sum( output ))

    # print('output.shape:', output.shape) # (1, N_COMPONENTS)

    # reshape and normalize:
    img_4 = np.reshape(img_4, (num_x, num_y))
    # print('img_4.min(), img_4.max()', img_4.min(), img_4.max())
    img_4 -= img_4.min()
    img_4 /= img_4.max()

    # visualize along with the previous step results:
    plt.subplot(1, 2, 1)
    plt.imshow(img_3, cmap='gray', extent=2*[min_val, max_val], interpolation='none')
    plt.title('Outputs of %d Manually Programmed RBFs with Random Exemplars' % N_COMPONENTS)
    plt.subplot(1, 2, 2)
    plt.title('Outputs of %d SciKit-Learn RBFSamplers with Random Exemplars' % N_COMPONENTS)
    plt.imshow(img_4, cmap='gray', extent=2*[min_val, max_val], interpolation='none')
    plt.suptitle('gamma = %.1f' % GAMMA)
    plt.show()

    # the images should look pretty similar provided that the centers are selected
    # randomly and differ for each method
    

    # 结论：
    # SciKit-Learn RBFSampler is claimed to approximate Radial Basis Function.
    # The experimental results show that approximation for 2D is not bad
    # (try different GAMMA and N_COMPONENTS).
    


if __name__ == '__main__':
    main()
