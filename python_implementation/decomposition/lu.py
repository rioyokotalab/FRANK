#!/bin/python
import numpy as np

from decomposition.utils.invertations import (
    invert_l_tri,
    invert_u_tri
)
from decomposition.utils.permutations import get_max_permutation_partial
from decomposition.utils.multiplications import (
    dense_dense,
    dense_sparse,
    sparse_dense,
    sparse_sparse
)
# from decomposition.utils.dmat import DMat
from decomposition.svd import rsvd


def update_l_u(LU):
    """
    Function handling the updating of L and U for lup_factorize.

    Both L and U are stored in LU (see lup_factorize).

    Arguments
    ---------
    L - np.array
    U - np.array
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


# TODO Calculate block size based on hardware.
def block_lu_factorize(LU, block_s):
    """
    Block LU decomposition with naive blocks.

    Arguments
    ---------
    LU - np.array
        Matrix for which block decomposition is to be calculated
    block_s - int
        Size of a block of the matrix.
    """
    # Check if input is square matrix and block_s matches
    m, n = LU.shape
    assert m == n, 'Input matrix is not a square matrix!'

    # Exit if remaining a is zero matrix
    if not LU.any():
        return 0

    # If matrix still has enough size for block LU, do it and recurse
    # Otherwise, finish up with normal LU
    if m > block_s:
        lup_factorize(LU[:block_s, :block_s])
        # NOTE Think about better storage
        # LU must not be inverted in place though...
        L = np.tril(LU[:block_s, :block_s], -1)\
            + np.identity(block_s)
        U = np.triu(LU[:block_s, :block_s])
        invert_l_tri(L)
        invert_u_tri(U)

        # Compute row
        LU[:block_s, block_s:] = L @ LU[:block_s, block_s:]
        # Compute column
        LU[block_s:, :block_s] = LU[block_s:, :block_s] @ U
        # Modify remaining block
        LU[block_s:, block_s:] -= (
            LU[block_s:, :block_s] @ LU[:block_s, block_s:]
        )
        # Recurse
        block_lu_factorize(LU[block_s:, block_s:], block_s)
    elif m <= block_s and m != 0:
        lup_factorize(LU)
    return 0


def rsvd_off_tridiagonal(arr_b, k):
    """
    Subroutine for sparse_block_lu, applying rsvd to off triagonal blocks.

    Arguments
    ---------
    arr_b - array
        Array of arrays containing matrix blocks.
        |A B|
        |C D| becomes [[A, B], [C, D]]
    k - int
        Target rank for the rsvd.
    """
    m = len(arr_b)
    for i in range(0, m):
        # Upper triangle
        for j in range(i+2, m):
            arr_b[i][j] = rsvd(arr_b[i][j], k, 5)
        # Lower triangle
        for j in range(0, i-1):
            arr_b[i][j] = rsvd(arr_b[i][j], k, 5)


def calc_off_tridiagonal(arr_b):
    """
    Explicitly calculate the off-tridiagonal compressed blocks.

    Arguments
    ---------
    arr_b - array
        Array of arrays containing matrix blocks.
        |A B|
        |C D| becomes [[A, B], [C, D]]
    """
    m = len(arr_b)
    for i in range(0, m):
        # Upper triangle
        for j in range(i+2, m):
            arr_b[i][j] = arr_b[i][j][0] @ arr_b[i][j][1] @ arr_b[i][j][2]
        # Lower triangle
        for j in range(0, i-1):
            arr_b[i][j] = arr_b[i][j][0] @ arr_b[i][j][1] @ arr_b[i][j][2]


def sparse_block_lu(arr_b, k):
    """
    Block LU decomposition with sparse off-tridiagonal blocks.

    Arguments
    ---------
    arr_b - array
        Array of arrays containing matrix blocks.
        |A B|
        |C D| becomes [[A, B], [C, D]]
        The off diagonal blocks must already be in sparse representation.
    block_s - int
        Size of a block of the matrix.
    k - int
        Rank with which off-tridiagonal blocks are approximized.
    """
    m = len(arr_b)
    n = len(arr_b[0])
    assert m == n
    # Exit if remaining a is zero matrix
    if not arr_b[0][0].any():
        return 0

    # If matrix still has enough size for block LU, do it and recurse
    # Otherwise, finish up with normal LU
    for l in range(0, m - 1):
        lup_factorize(arr_b[l][l])
        # NOTE Think about better storage
        # LU must not be inverted in place though...
        L = np.tril(arr_b[l][l], -1) + np.identity(arr_b[l][l].shape[0])
        U = np.triu(arr_b[l][l])
        invert_l_tri(L)
        invert_u_tri(U)

        # Compute dense first off-diag block of first row
        arr_b[l][l+1] = dense_dense(L, arr_b[l][l+1])
        # And rest of first row
        for i in range(l+2, m):
            arr_b[l][i] = rsvd(dense_sparse(L, arr_b[l][i]), k, 5)
        # Compute dense first block of column
        arr_b[l+1][l] = dense_dense(arr_b[l+1][l], U)
        # And rest of first column
        for i in range(l+2, n):
            arr_b[i][l] = rsvd(sparse_dense(arr_b[i][l], U), k, 5)

        # Modify remaining block (which will be recursed into)
        # Top left block
        arr_b[l+1][l+1] -= dense_dense(arr_b[l+1][l], arr_b[l][l+1])
        # Off-diagonal dense block in second row
        if l < n - 2:
            arr_b[l+1][l+2] -= dense_sparse(arr_b[l+1][l], arr_b[l][l+2])
        # Second row
        for i in range(l+3, n):
            arr_b[l+1][i] = rsvd(
                arr_b[l+1][i][0] @ arr_b[l+1][i][1] @ arr_b[l+1][i][2]
                - dense_sparse(arr_b[l+1][l], arr_b[l][i]),
                k,
                5
            )
        # Off-diagonal dense block in second column
        if l < m - 2:
            arr_b[l+2][l+1] -= sparse_dense(arr_b[l+2][l], arr_b[l][l+1])
        for i in range(l+3, m):
            arr_b[i][l+1] = rsvd(
                arr_b[i][l+1][0] @ arr_b[i][l+1][1] @ arr_b[i][l+1][2]
                - sparse_dense(arr_b[i][l], arr_b[l][l+1]),
                k,
                5
            )
        # Rest of the bottom block
        for i in range(l+2, m):
            for j in range(l+2, n):
                # Sparse target blocks
                if abs(i - j) > 1:
                    arr_b[i][j] = rsvd(
                        arr_b[i][j][0] @ arr_b[i][j][1] @ arr_b[i][j][2]
                        - sparse_sparse(arr_b[i][l], arr_b[l][j]),
                        k,
                        5
                    )
                # Dense target blocks
                else:
                    arr_b[i][j] = (
                        arr_b[i][j] - sparse_sparse(arr_b[i][l], arr_b[l][j])
                    )
    # Finally do the last diagonal block
    lup_factorize(arr_b[-1][-1])
    return 0


def h_sparse_lu(h_arr, k, p):
    """
    Compute the hierarchical sparse LU decomposition of the input array.
    """
    # Recurse until max depth is reached
    if isinstance(h_arr[0, 0], DMat):
        lup_factorize(h_arr[0, 0].arr)
    else:
        h_sparse_lu(h_arr[0, 0], k, p)
    print("getrf() result\n", h_arr[0, 0].get_dense(), "\nend\n")
    # Update rest of matrix
    h_arr[0, 1] = h_arr[0, 0].lower_trsm(h_arr[0, 1])
    h_arr[1, 0] = h_arr[0, 0].upper_trsm(h_arr[1, 0])
    h_arr[1, 1] -= h_arr[1, 0] @ h_arr[0, 1]
    # Finally decompose bottom-right
    if isinstance(h_arr[0, 0], DMat):
        lup_factorize(h_arr[1, 1].arr)
    else:
        h_sparse_lu(h_arr[1, 1], k, p)
    print("getrf() result\n", h_arr[1, 1].get_dense(), "\nend\n")
