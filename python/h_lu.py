from hicma.hierarchical import Hierarchical
from hicma.dense import Dense
from hicma.functions import (
    laplace1d,
    rand_data,
    zeros,
    Rand48
)

import numpy as np
import sys

def main():
    np.set_printoptions(formatter={'float': '{: 0.15f}'.format}, linewidth=100)
    N = 256
    nleaf = 16
    rank = 8
    assert len(sys.argv) == 2
    nblocks = 0
    admis = 0
    randx = []
    rnd = Rand48(0)
    for i in range(N):
        randx.append(rnd.drand())
    randx = np.array(sorted(randx))
    if (int(sys.argv[1]) == 0):
        # 1 level, full rank
        nblocks = int(N / nleaf)
        admis = int(N / nleaf)
    elif (int(sys.argv[1]) == 1):
        # log_2(N/nleaf) levels, full rank
        nblocks = 2
        admis = int(N / nleaf)
    elif (int(sys.argv[1]) == 2):
        # log_4(N/nleaf) levels, full rank
        nblocks = 4
        admis = int(N / nleaf)
    elif (int(sys.argv[1]) == 3):
        # 1 level, weak admissibility
        nblocks = int(N / nleaf)
        admis = 0
    elif (int(sys.argv[1]) == 4):
        # 1 level, strong admissibility
        nblocks = int(N / nleaf)
        admis = 1
    elif (int(sys.argv[1]) == 5):
        # log_2(N/nleaf) levels, weak admissibility
        nblocks = 2
        admis = 0
    elif (int(sys.argv[1]) == 6):
        # log_2(N/nleaf) levels, strong admissibility
        nblocks = 2
        admis = 1
    A = Hierarchical(randx, laplace1d, N, N, rank, nleaf, admis, nblocks, nblocks)
    x = Hierarchical(randx, rand_data, N, 1, rank, nleaf, admis, nblocks, 1)
    b = Hierarchical(randx, zeros, N, 1, rank, nleaf, admis, nblocks, 1)
    b.gemm(A, x)
    A.getrf()
    print(Dense(A).data)
    b.trsm(A, 'l')
    b.trsm(A, 'u')
    diff = (Dense(x) + Dense(b)).norm()
    norm = x.norm()
    print('Rel. L2 Error: {}'.format(np.sqrt(diff/norm)))


if __name__ == '__main__':
    main()
