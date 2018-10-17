#!/bin/python
import numpy as np


def get_max_permutation_partial(arr):
    """
    Find partial max permutation for the (0, 0) element of the input array.

    Attempt to find the absolute maximum element in the 0th column of `arr`.
    If the column is entirely 0, search for the maximum in the entire array.
    Finally, return the permutation matrices to bring the element into the
    position (0, 0) by switching rows and potentially columns.

    Arguments
    ---------
    arr : np.array
        Array for which the permutations are to be found.

    Returns
    -------
    permute : bool
        Wether or not a permutation is needed. Used to safe computation time.
    Pr : np.array
        The permutation matrix for the rows. A mxm square matrix where m is the
        number of rows of the input matrix.
    Pc : np.array
        The permutation matrix for the columns. A nxn square matrix where n is
        the number of columns of the input matrix.
    """
    Pr = np.identity(arr.shape[0])
    Pc = np.identity(arr.shape[1])
    permute = False
    row_max = np.argmax(np.absolute(arr[:, 0]))
    if arr[0, 0] == 0:
        permute = True
        # If column is entirely 0, include column switching
        if row_max == 0:
            row_max, col_max = np.unravel_index(
                np.argmax(np.absolute(arr)),
                arr.shape
            )
            Pc[:, [0, col_max]] = Pc[:, [col_max, 0]]
        Pr[[0, row_max], :] = Pr[[row_max, 0], :]
    return permute, Pr, Pc
