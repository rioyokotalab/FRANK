#!/bin/python
import numpy as np

from decomposition.qr import (
    house_pivot_qr,
    get_q,
    get_r
)
from decomposition.utils import (
    gen_matrix,
    invertations
)

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


def main():
    n = 5
    # Ints for better visual check
    # Green matrix for more realism
    array = gen_matrix.green(n)
    colnorm = np.array([np.dot(array[:, i], array[:, i]) for i in range(0, n)])
    perm = np.arange(n)

    print('Input array:\n', array)
    house_pivot_qr(array, colnorm, perm)
    Q = get_q(array)
    R = get_r(array)
    # Invert the permutation
    perm_inv = invertations.invert_perm(perm)
    # Check the result
    A = np.empty((n, n))
    A = Q @ R[:, perm_inv]
    print('Result:\n', array)
    print('Control:\n', A)


if __name__ == '__main__':
    main()
