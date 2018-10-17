#!/bin/python
import numpy as np

from decomposition.qr import (
    house_pivot_qr,
    get_q,
    get_r
)
from decomposition.utils import invertations


def one_sided_id(arr, k, perm):
    """
    Compute the one-sided ID of the input array for k ranks.

    arr is modified in place to contain
    |C1 T    |
    |C2 arr22|
    where arr22 is the part not done during the QR decomposition that is part
    of this algorithm.
    C and T can be extracted with functions defined for that purpose.

    Arguments
    ---------
    arr - np.array
        Input array that will be modified in place to contain C and V.
    k - int
        Rank for which the one sided ID is to be computed.
    perm - np.array
        Vector to store the permutations done during the QR part of the
        algorithm.
    """
    n = arr.shape[0]
    colnorm = np.array([np.dot(arr[:, i], arr[:, i]) for i in range(0, n)])
    # Do partial column pivoted QR and retrieve results
    house_pivot_qr(arr, colnorm, perm, k)
    Q = get_q(arr)
    R = get_r(arr)
    # Compute the ID matrices
    arr[:, :k] = Q[:, :k] @ R[:k, :k]
    # Solve for T
    invertations.invert_u_tri(R[:k, :k])
    arr[:k, k:] = R[:k, :k] @ R[:k, k:]


def get_c(arr, k):
    """
    Get the C matrix from a result array of one_sided_id performed with rank k.

    Arguments
    ---------
    arr - np.array
        Output array of the one_sided_id algorithm.
    k - int
        Rank that one_sided_id was performed with

    Returns
    -------
     - np.array
        The array C that was stored in arr.
    """
    return arr[:, :k]


def get_v(arr, k, perm):
    """
    Get the V matrix from a result array of one_sided_id performed with rank k.

    Arguments
    ---------
    arr - np.array
        Output array of the one_sided_id algorithm.
    k - int
        Rank that one_sided_id was performed with
    perm - np.array
        Permutation array resulting from one_sided_id.

    Returns
    -------
    V - np.array
        The V array from the result of one_sided_id.
    """
    m, n = arr.shape
    V = np.empty((k, n))
    V[:, perm] = np.c_[np.identity(k), arr[:k, k:]]
    return V
