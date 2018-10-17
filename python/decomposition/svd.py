#!/bin/python
import numpy as np
import scipy.linalg as sl

# from decomposition.qr import (
#     house_qr,
#     get_q,
#     get_r
# )


def golub_kahan_step(arr):
    """

    """
    pass


def svd(arr, k):
    """
    Compute the Singular Value Decomposition of arr to rank k


    Arguments
    ---------
    arr - np.array
        Input matrix that will be overwritten with the result.

    k - int
        Rank until which the SVD is to be computed
    """
    m, n = arr.shape
    assert m >= n
    pass


def rsvd(arr, k, p):
    """
    Compute the randomized SVD

    Arguments
    ---------
    arr - np.array
    k - int
        Target rank
    p - int
        Oversampling
    """
    m, n = arr.shape
    G = np.random.randn(n, k+p)
    Y = arr @ G
    # house_qr(Y)
    # Q = get_q(Y)
    # R = get_r(Y)
    Q, R = sl.qr(Y, pivoting=False, mode='economic')
    B = Q.transpose() @ arr
    Uhat, s, V = sl.svd(B)
    # Create truncated output matrices
    U = (Q @ Uhat)[:, :k]
    D = np.diag(s[:k])
    V = V[:k, :]
    return U, D, V
