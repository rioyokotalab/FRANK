#!/bin/python
import numpy as np

from decomposition.lu import block_lu_factorize

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


def main():
    n = 250
    block_size = 4
    array = np.random.random_integers(-10, 10, (n, n)).astype(float)
    test = np.copy(array)
    print('Input array:\n', array)
    block_lu_factorize(array, block_size)
    L = np.tril(array, -1) + np.identity(n)
    U = np.triu(array)
    print('Result:\n', array)
    print('Check:\n', L @ U)
    print(
        'Is result correct? :\n',
        np.allclose(L @ U, test)
    )


if __name__ == '__main__':
    main()
