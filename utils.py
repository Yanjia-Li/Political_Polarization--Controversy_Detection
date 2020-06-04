

import numpy as np
import scipy.sparse as sp

def normalization(adjacency):

    """L=D^-0.5*(A+I)*D^-0.5"""

    # Sparse matrix with ones on diagonal
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    # Construct a sparse matrix from diagonals.
    d_hat = sp.diags(np.power(degree, -0.5).flatten())

    return d_hat.dot(adjacency).dot(d_hat).tocoo()















