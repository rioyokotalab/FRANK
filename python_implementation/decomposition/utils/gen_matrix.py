#!/bin/python
import numpy as np


def green(n):
    """
    Generates a (n, n) matrix based on the 3D Greens function.

    As the shape of the matrix, not the dimensions matter, distances are drawn
    from randomized numbers in 1D.

    Arguments
    ---------
    n - int
        Number of rows and columns of generated matrix.
    """
    x = np.random.random([n, 1])
    arr = np.identity(n)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            arr[i, j] = 1./(4*np.pi*np.abs(x[i] - x[j]))
    return arr


def green_mesh(x):
    """
    (Hopefully) faster implementation of green using numpy.

    Arguments
    ---------
    x - np.array
        Vector of positions
    """
    n = len(x)
    mesh = np.vstack([x]*n)
    abs_diff = np.abs(mesh - mesh.transpose())
    # Avoid division by zero
    # np.fill_diagonal(abs_diff, 1)
    abs_diff += 0.001
    arr = np.reciprocal(abs_diff)
    # np.fill_diagonal(arr, 4)
    return arr
