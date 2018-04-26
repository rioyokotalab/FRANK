#!/bin/python
import numpy as np
import time

from decomposition.utils.gen_matrix import green_mesh
from decomposition.utils.hmat import HMat
# from decomposition.lu import h_sparse_lu

np.set_printoptions(
    formatter={'float': '{: 0.15f}'.format},
    suppress=True,
    linewidth=np.nan,
    threshold=np.nan
)

def main():
    n = 2**3
    max_n_leaf = 2
    k = 8
    p = 1
    # Generate random positions
    # block_s = int(n/max_n_leaf)
    # n_blocks = int(n/block_s)
    # x = np.hstack([
    #         (np.random.random(block_s).astype(float)*0.3 + i)
    #         for i in range(0, n_blocks)
    # ])
    t0 = time.time()
    seed = np.arange(n).astype(np.float64)
    arr = green_mesh(seed)
    arr_check = np.copy(arr)
    A = HMat(arr=arr, max_n_leaf=max_n_leaf, k=k, p=p)
    t1 = time.time()
    init_time = t1 - t0
    print("{:25} {:.6f}".format("Init HMat time:", init_time))
    # TODO Make below stuff work
    # x = HMat(
    #     arr=np.arange(n).astype(np.float64).reshape((n, 1)),
    #     max_n_leaf=max_n_leaf,
    #     k=k,
    #     p=p,
    #     nj_level=1
    # )
    # b = HMat(
    #     arr=np.zeros((n, 1)).astype(np.float64),
    #     max_n_leaf=max_n_leaf,
    #     k=k,
    #     p=p,
    #     nj_level=1
    # )
    # b -= A @ x
    x = np.arange(n).astype(np.float64).reshape(n, 1)
    b = A.get_dense() @ x

    t0 = time.time()
    A.getrf()
    t1 = time.time()
    exec_time = t1 - t0
    print("{:25} {:.6f}".format("Exec LU time: ", exec_time))

    # Extract L and U from array
    t0 = time.time()
    test = A.reconstruct_from_lu()
    t1 = time.time()
    reco_time = t1 - t0
    print("{:25} {:.6f}".format("Reconstruction time: ", reco_time))
    error = (
        np.linalg.norm(test - arr_check)
        / np.linalg.norm(arr_check)
    )
    print("{:25} {:1.2e}".format("Relative error: ", error))

    # Forward substitution
    b = np.linalg.inv(np.tril(A.get_dense(), -1) + np.eye(n)) @ b
    # Backward substitution
    b = np.linalg.inv(np.triu(A.get_dense())) @ b
    print(b-x)
    error = (np.linalg.norm(b-x) / np.linalg.norm(x))
    print("{:25} {:1.2e}".format("Relative error: ", error))


if __name__ == '__main__':
    main()
