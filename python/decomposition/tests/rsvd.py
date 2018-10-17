#!/bin/python
import numpy as np
from scipy.linalg import hilbert

from decomposition.svd import rsvd
# from decomposition.utils import gen_matrix

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


def main():
    n = 50
    for i in range(1, 15):
        # arr = gen_matrix.green(n)
        arr = hilbert(n)
        arr_check = np.copy(arr)
        U, D, V = rsvd(arr, i, 5)
        print(np.linalg.norm(U @ D @ V - arr_check)/np.linalg.norm(arr_check))


if __name__ == '__main__':
    main()
