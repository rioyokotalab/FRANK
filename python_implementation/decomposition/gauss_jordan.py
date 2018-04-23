#!/bin/python
import numpy as np

from decomposition.utils.permutations import get_max_permutation_partial


def gauss_jordan(arr):
    """
    Takes an augmented matrix and applied the Gauss-Jordan algorithm.

    Uses the absolute maximum element of the remaining matrix, eg A in
    |1 v|
    |0 A|
    as the pivot by switching rows and colums.

    Arguments
    ---------
    arr - np.array of shape (n, m)
        Augmented input matrix. This is the full equation that is to be solved.
    """
    m, n = arr.shape
    for i in range(0, m):
        # Find permutation if necessary
        Pr = np.identity(m)
        Pc = np.identity(n)
        permute, Pr[i:, i:m], Pc[i:m, i:m] = get_max_permutation_partial(
                arr[i:, i:n-1]
        )
        # Permute if necessary
        if permute:
            arr = Pr @ arr @ Pc
        pivot = arr[i, i]
        if pivot == 0:
            break
        # Construct the Gauss-Jordan matrix
        GJ = np.zeros([m, m])
        GJ[0:i, 0:i] = np.identity(i)
        GJ[0:i, i] = -1/pivot * arr[0:i, i]
        GJ[i, i] = 1/pivot
        GJ[i+1:, i] = -1/pivot * arr[i+1:, i]
        GJ[i+1:, i+1:] = np.identity(m-i)[:arr.shape[0]-i-1, :m-i-1]
        # Transform array
        arr = np.dot(GJ, arr)
    return arr


equations = np.array([
    [2.0, 1.0, -1.0],
    [-3.0, -1.0, 2.0],
    [-2.0, 1.0, 2.0]
])
solution = np.array([8.0, -11.0, -3.0])
augmented = np.c_[equations, solution]
print('Augmented input matrix: \n', augmented)
result = gauss_jordan(augmented)
print('Result of GJ algprithm: \n', result)
print('Result from numpy: ', np.linalg.solve(equations, solution))
