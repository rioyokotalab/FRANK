#!/bin/python
import numpy as np

from decomposition.lu import lup_factorize

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


def main():
    array = np.array([
        [-5, 1, -3, 4],
        [8, -7, 3, 2],
        [-3, -6, -1, -1],
        [0, 0, 3, 9]
        ],
        dtype=np.float32
    )
    Pr = np.identity(array.shape[0])
    Pc = np.identity(array.shape[1])

    print('Input array: \n', array, '\n')
    lup_factorize(array, Pr, Pc)
    # Extract L and U from array
    L = np.tril(array, -1) + np.identity(array.shape[0])
    U = np.triu(array, 0)
    print(
        'Result LUP: \nL:\n', L,
        '\nU:\n', U,
        '\nRecombined:\n', Pr @ L @ U @ Pc
    )


if __name__ == '__main__':
    main()
