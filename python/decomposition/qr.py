#!/bin/python
import numpy as np

from decomposition.utils.householder import get_vector


def house_qr(arr):
    """
    Perform householder QR on the arr, saving the results in its place.

    General shape of the result:
    |R11 R12 R13|
    |v12 R22 R23|
    |v13 v23 R33|
    This means that the relevant part of the Householder vector is saved in
    the lower triangle of the result (without the diagonal) and the upper
    triangle holds the R part.

    Arguments
    ---------
    arr - np.array
        Matrix to be decomposited.
    """
    m, n = arr.shape
    if m > 1:
        v, beta = get_vector(arr[:, 0])
        # Apply householder rotation
        arr[:, :] = (np.identity(m) - beta * np.outer(v, v)) @ arr[:, :]
        # Save householder vector in arr
        arr[1:, 0] = v[1:]
        # Recurse
        house_qr(arr[1:, 1:])
        return 0
    else:
        return 1


def house_pivot_qr(arr, colnorm, perm, k=None):
    """
    Performs Householder QR with pivoting.

    For details on how data is stored in arr, see house_qr.


    Arguments
    ---------
    arr - np.array
    colnorm - np.array
        Vector holding the column norms of arr. Updated during the algorithm.
    perm - np.array
        Vector holding the permutations. Updated during the algorithm
    k - int
        Rank until which to compute the QR. If not provided or None, a full
        column pivot QR will be computed.
    """
    m, n = arr.shape
    if k is None:
        k = m
    for i in range(0, k):
        # Find column with max abs value and permute if necessary
        p = np.argmax(colnorm[i:]) + i
        if colnorm[p] == 0:
            break
        if p != i:
            perm[i], perm[p] = perm[p], perm[i]
            arr[:, [i, p]] = arr[:, [p, i]]
            colnorm[i], colnorm[p] = colnorm[p], colnorm[i]
        v, beta = get_vector(arr[i:, i])
        arr[i:, i:] = (
            (np.identity(m-i) - beta * np.outer(v, v)) @ arr[i:, i:]
        )
        arr[i+1:, i] = v[1:]
        # Update colnorm
        colnorm[i+1:m] = colnorm[i+1:m] - arr[i, i+1:m]**2


def house_bidiag(arr):
    """
    Bring the input array into bidiagonalized form using Householder matrizes.

    Arguments
    ---------
    arr - np.array
    """
    m, n = arr.shape
    v, beta = get_vector(arr[:, 0])
    arr[:, :] = (np.identity(m) - beta*np.outer(v, v)) @ arr[:, :]
    # arr[1:, 0] = v[1:]
    if n > 1:
        v, beta = get_vector(arr[0, 1:])
        arr[:, 1:] = arr[:, 1:] @ (np.identity(n - 1) - beta*np.outer(v, v))
        # arr[j, 2:] = v[1:]
        house_bidiag(arr[1:, 1:])


def house_triag(arr):
    """
    Bring the input array into tridiagonalized form using Householder matrizes.

    Arguments
    --------
    arr - np.array
    """


def get_q(arr):
    """
    Get the Q matrix from the combined result of a qr.

    A = Q.R, or A.P = Q.R if permutation were used.

    Arguments
    ---------
    arr - np.array
        Output array of a qr decomposition containing both the Householder
        vectors and the upper triangle matrix R.
    perm - np.array
        Vector containing the permutation operations.

    Returns
    -------
    Q - np.array
    """
    n = arr.shape[0]
    # Generate Q explicitly
    Q = np.identity(n)
    H = np.identity(n) + np.tril(arr, -1)
    for i in range(0, n-1):
        v = H[i:, i]
        beta = 2./np.dot(v, v)
        Q_mul = np.identity(n-i) - beta * np.outer(v, v)
        Q[i:, :] = (
            np.dot(Q_mul, Q[i:, :])
        )
    return Q.transpose()


def get_r(arr):
    """
    Get the R matrix from the combined result of a qr.

    Simple function to make core more readable.

    Arguments
    ---------
    arr - np.array
        Output array of a qr decomposition containing both the Householder
        vectors and the upper triangle matrix R.

    Returns
    -------
    R - np.array
        Upper right triangle of the input matrix
    """
    return np.triu(arr)


def get_p(perm):
    """
    Create a permutation matrix from a vector of permutations.

    Arguments
    ---------
    perm - np.array
        Vector containing the permutations.

    Returns
    -------
    P - np.array
        The permutation matrix created from perm.
    """
    n = perm.shape[0]
    P = np.zeros((n, n))
    for i in range(0, n):
        P[perm[i], i] = 1
    return P
