#!/bin/python
import numpy as np
from scipy.linalg import hilbert

from decomposition.qr import (
    house_bidiag
)
from decomposition.utils import (
    gen_matrix
)

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


def main():
    n = 5
    # Ints for better visual check
    # Green matrix for more realism
    array = gen_matrix.green(n)
    array = hilbert(n)

    print('Input array:\n', array)
    house_bidiag(array)
    # Check the result
    print('Result:\n', array)


if __name__ == '__main__':
    main()
