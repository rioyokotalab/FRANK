import numpy as np
import sys

from FRANK.dense import Dense
from FRANK.low_rank import LowRank
from FRANK.hierarchical import Hierarchical
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
    N = 256
    nleaf = 16
    nblocks = int(N / nleaf)
    rank = 8
    admis = 1
    randx = []
    rnd = Rand48(0)
    for i in range(N):
        randx.append(rnd.drand())
    randx = np.array(sorted(randx))
    x = Hierarchical(ni=nblocks)
    b = Hierarchical(ni=nblocks)
    A = Hierarchical(ni=nblocks, nj=nblocks)
    D = Hierarchical(ni=nblocks, nj=nblocks)
    Ubase = [None] * nblocks
    Vbase = [None] * nblocks
    start('Init Matrix')
    for ib in range(nblocks):
        xi = Dense(ni=nleaf)
        bj = Dense(ni=nleaf)
        for il in range(nleaf):
            xi[il] = randx[ib*nleaf+il]
            bj[il] = 0
        x[ib] = xi
        b[ib] = bj
    for ib in range(nblocks):
        for jb in range(nblocks):
            Aij = Dense(randx, laplace1d, nleaf, nleaf, nleaf*ib, nleaf*jb)
            D[ib, jb] = Aij
            if abs(ib - jb) <= admis:
                A[ib, jb] = Aij
            else:
                # If one of the bases is not none yet, do SVD.
                # The order (ib, jb) is important for efficiency here.
                # With the current loop order, things are not very efficient.
                if Ubase[ib] is None or Vbase[jb] is None:
                    A[ib, jb] = LowRank(Aij, k=rank)
                    Ubase[ib] = A[ib, jb].U
                    Vbase[jb] = A[ib, jb].V
                # If both are known, create LowRank from them and calculate
                # S afterwards
                else:
                    A[ib, jb] = LowRank(m=nleaf, n=nleaf, k=rank)
                    A[ib, jb].U = Ubase[ib]
                    A[ib, jb].V = Vbase[jb]
                    # Note that we multiply with transpose of U, thus dim[1]
                    UtxA = Dense(ni=Ubase[ib].dim[1], nj=Aij.dim[1])
                    UtxA.gemm_trans(Ubase[ib], Aij, True, False, 1, 0)
                    A[ib, jb].S.gemm_trans(UtxA, Vbase[jb], False, True, 1, 0)
    diff = (Dense(A) - Dense(D)).norm()
    norm = D.norm()
    printf('Compression Accuracy')
    printf('Rel. L2 Error', np.sqrt(diff/norm))
    printf('Time')
    b.gemm(A, x)
    stop('Init Matrix')
    start('LU decomposition')
    for ib in range(nblocks):
        A[ib, ib].getrf()
        for jb in range(ib+1, nblocks):
            A[ib, jb].trsm(A[ib, ib], 'l')
            A[jb, ib].trsm(A[ib, ib], 'u')
        for jb in range(ib+1, nblocks):
            for kb in range(ib+1, nblocks):
                A[jb, kb].gemm(A[jb, ib], A[ib, kb])
    stop('LU decomposition')
    start('Forward substitution')
    for ib in range(nblocks):
        for jb in range(ib):
            b[ib].gemm(A[ib, jb], b[jb])
        b[ib].trsm(A[ib, ib], 'l')
    stop('Forward substitution')
    start('Backward substitution')
    for ib in range(nblocks-1, -1, -1):
        for jb in range(nblocks-1, ib, -1):
            b[ib].gemm(A[ib, jb], b[jb])
        b[ib].trsm(A[ib, ib], 'u')
    stop('Backward substitution')
    diff = (Dense(x) + Dense(b)).norm()
    norm = x.norm()
    printf('LU Accuracy')
    printf('Rel. L2 Error', np.sqrt(diff/norm))


if __name__ == '__main__':
    main()
