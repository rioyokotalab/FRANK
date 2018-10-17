#!/bin/python
import numpy as np


def invert_l_tri(arr):
    """
    Inverts lower triangular matrix arr in place.

    Arguments
    --------
    arr - np.array
        Lower triangular input array that will be inverted in place.
    """
    m, n = arr.shape
    # Adapt diagonal elements
    for i in range(0, m):
        arr[i, i] = 1/arr[i, i]
    # Calculate lower triangle
    # Note that since the diagonal is already inverted, the division by
    # diagonal elements of arr changes to a multiplication
    for i in range(1, m):
        for j in range(0, i):
            arr[i, j] = - np.dot(arr[i, j:i], arr[j:i, j]) * arr[i, i]


def invert_u_tri(arr):
    """
    Inverts upper triangular matrix arr in place.

    Arguments
    --------
    arr - np.array
        Upper triangular input array that will be inverted in place.
    """
    m, n = arr.shape
    # Adapt diagonal elements
    for i in range(0, m):
        arr[i, i] = 1/arr[i, i]
    # Calculate lower triangle
    # Note that since the diagonal is already inverted, the division by
    # diagonal elements of arr changes to a multiplication
    for i in range(m-2, -1, -1):
        for j in range(m-1, i, -1):
            arr[i, j] = - np.dot(arr[i, i+1:j+1], arr[i+1:j+1, j]) * arr[i, i]


def invert_perm(perm):
    """
    Invert a permutation vector.

    Let P be the permutation. If A.P = Q then A[:, perm] = Q and A = Q[:, inv].

    Arguments
    ---------
    perm - np.array
        Permutation vector.

    Returns
    -------
    inv - np.array
        The inverted permutation vector.
    """
    inv = np.empty(perm.shape, dtype=int)
    inv[perm] = np.arange(perm.shape[0])
    return inv
