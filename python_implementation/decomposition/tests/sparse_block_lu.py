#!/bin/python
import numpy as np
import time
import matplotlib.pyplot as plt

from decomposition.utils.gen_matrix import (
    green_mesh
)
from decomposition.utils.matrix import make_block
from decomposition.lu import (
    sparse_block_lu,
    rsvd_off_tridiagonal,
    calc_off_tridiagonal
)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


def plot():
    n, k, block_s, error, exec_time = np.loadtxt(
        './benchmark/sparse_time',
        delimiter='\t',
        unpack=True,
        skiprows=1
    )
    # Plot data
    plt.plot(
        n,
        exec_time,
        label='Sparse LU'
    )
    # Plot N and N^2
    x = [i/2000 for i in n]
    plt.plot(n, x, label=r'$N$')
    x2 = [i**2/20000 for i in n]
    plt.plot(n, x2, label=r'$N^2$')

    plt.xscale('log', basex=2)
    plt.yscale('log', basey=10)

    plt.title('LU with sparse off-tridiagonal')
    plt.xlabel('Matrix size')
    plt.ylabel('Execution time [ms]')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    block_s = 128

    with open('./benchmark/sparse_time', 'w') as f:
        f.write(
            '{:>10}\t{:>10}\t{:>10}\t{:>15}\t{:>15}\n'.format(
                'N', 'Block Size', 'Rank', 'Error', 'Time [s]'
            )
        )
        for n_blocks in range(2, 40):
            n = n_blocks * block_s
            k = int(1./16 * block_s)
            arr = green_mesh(n, block_s)
            arr_check = np.copy(arr)
            # print('Input array:\n', array)
            arr_b = make_block(arr, block_s)
            rsvd_off_tridiagonal(arr_b, k)

            # Time execution
            t0 = time.time()
            sparse_block_lu(arr_b, k)
            t1 = time.time()
            # Get time in MS
            exec_time = (t1 - t0)
            print(n, exec_time)

            # Calculate precision
            calc_off_tridiagonal(arr_b)
            arr = np.block(arr_b)
            # Extract L and U from array
            L = np.tril(arr, -1) + np.identity(arr.shape[0])
            U = np.triu(arr, 0)
            error = (
                np.linalg.norm(L @ U - arr_check) / np.linalg.norm(arr_check)
            )
            f.write(
                '{:10}\t{:10}\t{:10}\t{:015.13e}\t{:015.8f}\n'.format(
                    n, block_s, k, error, exec_time
                )
            )
            f.flush()


if __name__ == '__main__':
    # main()
    plot()
