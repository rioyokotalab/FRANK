#!/bin/python


def dense_dense(arr1, arr2):
    """

    Arguments
    ---------
    arr1 - np.array
        Dense matrix.
    arr2 - np.array
        Dense matrix.
    """
    return arr1 @ arr2


def dense_sparse(arr1, arr2):
    """

    Arguments
    ---------
    arr1 - np.array
        Dense matrix.
    arr2 - np.array
        SVD decomposision stored as array of arrays. [U, D, V]
    """
    return ((arr1 @ arr2[0]) @ arr2[1]) @ arr2[2]


def sparse_dense(arr1, arr2):
    """

    Arguments
    ---------
    arr1 - np.array
        Dense matrix.
    arr2 - np.array
        SVD decomposision stored as array of arrays. [U, D, V]
    """
    return arr1[0] @ (arr1[1] @ (arr1[2] @ arr2))


def sparse_sparse(arr1, arr2):
    """

    Arguments
    ---------
    arr1 - np.array
        SVD decomposision stored as array of arrays. [U, D, V]
    arr2 - np.array
        SVD decomposision stored as array of arrays. [U, D, V]
    """
    return arr1[0] @ (arr1[1] @ (arr1[2] @ arr2[0]) @ arr2[1]) @ arr2[2]
