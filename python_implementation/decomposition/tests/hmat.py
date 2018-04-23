#!/bin/python
import numpy as np
from scipy import allclose

from decomposition.utils.gen_matrix import green_mesh
from decomposition.utils.matrix import (
    h_mat,
    ut_mat,
    lt_mat,
    make_block
)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


def test_mul(h_arr, test_b):
    # Multiplication tests
    for i in range(0, 2):
        for j in range(0, 2):
            A = h_arr[i, j]
            for p in range(0, 2):
                for q in range(0, 2):
                    B = h_arr[p, q]
                    h_test = (A @ B).get_dense_array()
                    d_test = test_b[i][j] @ test_b[p][q]
                    error = (
                        np.linalg.norm(h_test - d_test)
                        / np.linalg.norm(d_test)
                    )
                    close = allclose(h_test, d_test)
                    if not close:
                        print('\n', type(A), ' @ ', type(B))
                        print(error, close)


def test_sub(h_arr, test_b):
    for i in range(0, 2):
        for j in range(0, 2):
            A = h_arr[i, j]
            for p in range(0, 2):
                for q in range(0, 2):
                    B = h_arr[p, q]
                    print(A, ' @ ', B)
                    h_test = (A - B).get_dense_array()
                    d_test = test_b[i][j] - test_b[p][q]
                    if i == p and j == q:
                        error = (
                            np.linalg.norm(h_test - d_test)
                            / np.linalg.norm(d_test)
                        )
                    else:
                        error = 'nan'
                    print(error, allclose(h_test, d_test))


def test_add(h_arr, test_b):
    for i in range(0, 2):
        for j in range(0, 2):
            A = h_arr[i, j]
            for p in range(0, 2):
                for q in range(0, 2):
                    B = h_arr[p, q]
                    print(A, ' @ ', B)
                    h_test = (A + B).get_dense_array()
                    d_test = test_b[i][j] + test_b[p][q]
                    error = (
                        np.linalg.norm(h_test - d_test)
                        / np.linalg.norm(d_test)
                    )
                    print(error, allclose(h_test, d_test))


def test_tri(arr, n_levels, n, k, p):
    test = np.copy(arr)
    test_b = make_block(test, int(n/2))
    h_arr = h_mat(arr, n_levels)
    h_arr.lr_off_diag(k, p)
    tri = ut_mat(h_arr[0, 0], n_levels-1)
    h_test = (tri @ h_arr[0, 1]).get_dense_array()
    d_test = np.triu(test_b[0][0]) @ test_b[0][1]
    error = (
        np.linalg.norm(h_test - d_test)
        / np.linalg.norm(d_test)
    )
    print(error, allclose(h_test, d_test))


def test_inv(arr, n_levels, n, k, p):
    test = np.copy(arr)
    test_b = make_block(test, int(n/2))
    h_arr = h_mat(arr, n_levels)
    h_arr.lr_off_diag(k, p)
    # Upper triangle
    tri = ut_mat(h_arr[0, 0], n_levels-1)
    tri.invert()
    h_test = tri.get_dense_array() @ np.triu(test_b[0][0])
    error = (
        np.linalg.norm(h_test - np.identity(h_test.shape[0]))
        / np.linalg.norm(np.triu(test_b[0][0]))
    )
    print(error, allclose(h_test, np.identity(h_test.shape[0])))
    # Lower triangle
    tri = lt_mat(h_arr[0, 0], n_levels-1)
    tri.invert()
    h_test = (
        tri.get_dense_array()
        @ (np.tril(test_b[0][0], -1) + np.identity(test_b[0][0].shape[0]))
    )
    error = (
        np.linalg.norm(h_test - np.identity(h_test.shape[0]))
        / np.linalg.norm(
            np.tril(test_b[0][0], -1) + np.identity(test_b[0][0].shape[0])
        )
    )
    print(error, allclose(h_test, np.identity(h_test.shape[0])))


def main():
    n_levels = 2
    n = 256
    k = 10
    p = 5
    arr = green_mesh(n, int(n/(2**n_levels)))
    test = np.copy(arr)
    test_b = make_block(test, int(n/2))
    h_arr = h_mat(arr, n_levels)
    h_arr.lr_off_diag(k, p)

    test_mul(h_arr, test_b)


if __name__ == '__main__':
    main()
