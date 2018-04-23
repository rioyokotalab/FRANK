#!/bin/python
import numpy as np
import time

from decomposition.utils.gen_matrix import green_mesh
from decomposition.utils.hmat import HMat
from decomposition.lu import h_sparse_lu

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


def main():
    n = 2**4
    max_n_leaf = 8
    k = 8
    p = 1
    # Generate random positions
    # block_s = int(n/max_n_leaf)
    # n_blocks = int(n/block_s)
    # x = np.hstack([
    #         (np.random.random(block_s).astype(float)*0.3 + i)
    #         for i in range(0, n_blocks)
    # ])
    x = np.arange(n).astype(float)
    arr = green_mesh(x)
    print(arr)
    arr_check = np.copy(arr)
    t0 = time.time()
    h_arr = HMat(arr=arr, max_n_leaf=max_n_leaf, k=k, p=p)
    t1 = time.time()
    init_time = t1 - t0
    print("{:25} {:.6f}".format("Init HMat time:", init_time))
    t0 = time.time()
    h_sparse_lu(h_arr, k, p)
    t1 = time.time()
    exec_time = t1 - t0
    print("{:25} {:.6f}".format("Exec LU time: ", exec_time))
    # Extract L and U from array
    t0 = time.time()
    # out = h_arr.get_dense()
    # L = np.tril(out, -1) + np.identity(out.shape[0])
    # U = np.triu(out)
    # test = L @ U
    test = h_arr.reconstruct_from_lu()
    t1 = time.time()
    reco_time = t1 - t0
    print("{:25} {:.6f}".format("Reconstruction time: ", reco_time))
    error = (
        np.linalg.norm(test - arr_check)
        / np.linalg.norm(arr_check)
    )
    print("{:25} {:1.2e}".format("Relative error: ", error))


if __name__ == '__main__':
    main()
