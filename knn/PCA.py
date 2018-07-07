import numpy as np 
import matplotlib.pyplot as plt 


# Principal Component Analysis:

class PCA(object):

    def __init__(self, n_components=20, visualize=True):
        self.n_components = n_components
        self.visualize = visualize

    def fit(self, data): 
        
        N, D = data.shape
        # mean normalization:
        data = data - np.mean(data, axis=0)
        
        # calculate the covariance matrix
        C = data.T.dot(data) / N
        #C = np.cov(data.T) # DxD matrix

        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since C is symmetric, 
        # the performance gain is substantial
        evals, evecs = np.linalg.eig(C) 

        # sort eigenvalue in decreasing order:
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]

        # sort eigenvectors according to the same index
        evecs = evecs[:,idx] # (D x D) matrix
        
        # select the first k eigenvectors (k is desired dimension
        # of rescaled data array, or n_components)
        self.evecs = evecs[:, :self.n_components]  # (D x k) matrix
        

                   
    def transform(self, data, mode='rescale'):
        """
        Returns: data transformed in 2 dims/columns + regenerated original data
        pass in: data as 2D NumPy array
        """ 
        N, D = data.shape

        # carry out the transformation on the data using eigenvectors:        
        rescaled = np.dot(data, self.evecs)
        recovered = rescaled.dot(self.evecs.T)    
        
        # visualize some samples from recovered data
        if self.visualize:

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.subplots_adjust(hspace=1, wspace=1)

            n = np.random.randint(0, N)

            decomposed = recovered[n, :].reshape(28, 28).astype(np.float32)
            original = data[n, :].reshape(28, 28).astype(np.float32)

            ax1.imshow(255-decomposed, cmap='gray')
            ax1.set_title('Decomposed')
            ax2.imshow(255-original, cmap='gray')
            ax2.set_title('Original')

            plt.show() 

        if mode == 'rescale':
            return rescaled

        elif mode == 'recover':
            return recovered
