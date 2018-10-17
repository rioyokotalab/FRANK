#!/bin/python
import numpy as np


def get_vector(x):
    """
    Calculate the Householder vector and prefactor for the input vector x.

    The Householder matrix can then be build as
    H = I - beta * np.outer(v, v)

    Arguments
    ---------
    x - np.array
        Vector for which the householder matrix is to be computed.

    Returns
    -------
    v - np.array
        Householder vector.
    beta - float
        Factor used to build
    """
    beta = None
    sig = np.dot(x[1:], x[1:])
    v = np.append([1], x[1:])
    if sig == 0:
        beta = 0
    else:
        mu = np.sqrt(x[0]**2 + sig)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = - sig / (x[0] + mu)
        beta = 2. * v[0]**2 / (sig + v[0]**2)
        v = v/v[0]
    return v, beta
