#!/bin/python
import numpy as np
import scipy.linalg as sl


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


def update_l_u(LU):
    """
    Function handling the updating of L and U for lup_factorize.

    Both L and U are stored in LU (see lup_factorize).

    Arguments
    ---------
    LU - np.array
    """
    # Split up the input matrix according to the scheme
    # |alpha v     |
    # |u     U_next|
    # U_next will be recursed into from the main function
    alpha = LU[0, 0]
    u = LU[1:, 0]
    v = LU[0, 1:]

    # Store new part of L in LU, thereby modifying it (and thus u, see blow)
    LU[1:, 0] = u/alpha
    # Update U_next
    # This is the Schur's compliment.
    # Note that the variable u differs between the previous and the next line.
    # LU is modified in place and u simply points to a column segment of it.
    # Thus, the factor alpha must not be used again here!
    LU[1:, 1:] -= np.outer(u, v)


def lup_factorize(LU, Pr=None, Pc=None):
    """
    Recursively calculates the LU decomposition of the input array.

    P * arr = L * U

    L and U are both stored in the input array LU, where U is the upper
    triangle including the diagonal, whereas L can be obtained from LU by
    taking the lower triangle and adding an identity matrix of appropriate
    size.
    Example:
         |U11 U12 U13|          |1    0    0|       |LU11 LU21 LU31|
    LU = |L21 U22 U23|  =>  L = |LU21 1    0| , U = |0    LU22 LU32|
         |L31 L32 U33|          |LU31 LU32 1|       |0    0    LU33|

    Arguments
    ---------
    LU - np.array
        Input array. Will be modified to store both L and U.
    Pr - np.array
        Row permutation matrix applied to decompose the matrix. If given as
    Pc - np.array
        Column permutation matrix applied to decompose the matrix.
        This matrix may be the identity matrix if column permutation was not
        necessary.
    """
    # Check if input is square matrix
    m, n = LU.shape
    assert m == n, 'Input matrix is not a square matrix!'
    # Exit if remaining a is zero matrix
    if not LU.any():
        return 1

    if Pr is None:
        Pr = np.identity(m)
    else:
        assert Pr.shape == LU.shape
    if Pc is None:
        Pc = np.identity(m)
    else:
        assert Pc.shape == LU.shape

    # If pivot is 0, find maximum in current column of a_tilde
    # and switch rows. Switch columns if entire column is 0.
    permute, Pr_new, Pc_new = get_max_permutation_partial(LU)
    # Avoid unnecessaty matrix multiplications by only multiplying when
    # permutation is necessary.
    if permute:
        Pr[:, :] = Pr_new @ Pr
        Pc[:, :] = Pc @ Pc_new
        LU[:, :] = Pr_new @ LU @ Pc_new

    update_l_u(LU)
    lup_factorize(LU[1:, 1:], Pr[1:, 1:], Pc[1:, 1:])
    return 0
