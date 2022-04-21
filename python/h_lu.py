import numpy as np
import sys

from FRANK.hierarchical import Hierarchical
from FRANK.dense import Dense
from FRANK.functions import (
    laplace1d,
    rand_data,
    zeros,
    Rand48
)
from FRANK.print import printf
from FRANK.timer import (
    start,
    stop
)


def main():
    np.set_printoptions(formatter={'float': '{: 0.15f}'.format}, linewidth=100)
    N = 128
    nleaf = 16
    rank = 4
    assert len(sys.argv) == 2
    nblocks = 0
    admis = 0
    randx = []
    rnd = Rand48(0)
    for i in range(N):
        randx.append(rnd.drand())
    randx = np.array(sorted(randx))
    start('Init Matrix')
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
    admis = int(N / nleaf) # Full rank
    D = Hierarchical(randx, laplace1d, N, N, rank, nleaf, admis, nblocks, nblocks)
    x = Hierarchical(randx, rand_data, N, 1, rank, nleaf, admis, nblocks, 1)
    b = Hierarchical(randx, zeros, N, 1, rank, nleaf, admis, nblocks, 1)
    diff = (Dense(A) - Dense(D)).norm()
    norm = D.norm()
    printf('Compression Accuracy')
    printf('Rel. L2 Error', np.sqrt(diff/norm))
    printf('Time')
    b.gemm(A, x)
    stop('Init Matrix')
    start('LU decomposition')
    A.getrf()
    stop('LU decomposition')
    start('Forward substitution')
    b.trsm(A, 'l')
    stop('Forward substitution')
    start('Backward substitution')
    b.trsm(A, 'u')
    stop('Backward substitution')
    diff = (Dense(x) + Dense(b)).norm()
    norm = x.norm()
    printf('LU Accuracy')
    printf('Rel. L2 Error', np.sqrt(diff/norm))


if __name__ == '__main__':
    main()
