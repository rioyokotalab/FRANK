#!/bin/python
import numpy as np

from decomposition.id import (
    one_sided_id,
    get_c,
    get_v
)
from decomposition.utils import gen_matrix

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


def main():
    n = 60
    for i in range(1, 15):
        arr = gen_matrix.green_mesh(n, 2)
        arr_check = np.copy(arr)
        perm = np.arange(n)
        one_sided_id(arr, i, perm)
        C = get_c(arr, i)
        V = get_v(arr, i, perm)
        print(np.linalg.norm(C @ V - arr_check)/np.linalg.norm(arr_check))


if __name__ == '__main__':
    main()
