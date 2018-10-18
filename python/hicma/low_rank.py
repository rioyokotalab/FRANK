#!/bin/python
import numpy as np

from hicma.id import rsvd
from hicma.node import Node
import hicma.dense as HD
import hicma.hierarchical as HH


class LowRank(Node):
    """
    Class for low-rank-form matrices.
    """
    def __init__(
            self,
            arr=None,
            m=None,
            n=None,
            rank=None,
            i_abs=0,
            j_abs=0,
            level=0
    ):
        """
        Initialize from input data
        """
        if isinstance(arr, list):
            super().__init__(i_abs, j_abs, level)
            assert isinstance(m, int) and isinstance(n, int)
            assert isinstance(rank, int)
            self.dim = [m, n]
            self.rank = rank
            assert len(arr) == 3
            assert isinstance(arr[0], HD.Dense)
            assert isinstance(arr[1], HD.Dense)
            assert isinstance(arr[2], HD.Dense)
            assert arr[0].arr.shape[1] == arr[1].arr.shape[0]
            assert arr[1].arr.shape[1] == arr[2].arr.shape[0]
            self.U = arr[0]
            self.S = arr[1]
            self.V = arr[2]
        elif isinstance(arr, np.ndarray):
            super().__init__(i_abs, j_abs, level)
            assert isinstance(m, int) and isinstance(n, int)
            assert isinstance(rank, int)
            self.dim = [m, n]
            self.rank = rank
            U, S, V = rsvd(arr, self.rank, 0)
            self.U = HD.Dense(U)
            self.S = HD.Dense(D)
            self.V = HD.Dense(V)
        elif isinstance(arr, HD.Dense):
            super().__init__(arr.i_abs, arr.j_abs, arr.level)
            self.dim = arr.dim
            assert isinstance(rank, int)
            self.rank = rank
            U, S, V = rsvd(arr.data, self.rank, 0)
            self.U = HD.Dense(U)
            self.S = HD.Dense(S)
            self.V = HD.Dense(V)
        elif isinstance(arr, LowRank):
            super().__init__(arr.i_abs, arr.j_abs, arr.level)
            self.dim = arr.dim
            self.rank = arr.rank
            self.U = HD.Dense(arr.U)
            self.S = HD.Dense(arr.S)
            self.V = HD.Dense(arr.V)
        elif arr is None:
            super().__init__(i_abs, j_abs, level)
            assert isinstance(m, int) and isinstance(n, int)
            assert isinstance(rank, int)
            self.dim = [m, n]
            self.rank = rank
            self.U = HD.Dense(ni=m, nj=rank)
            self.S = HD.Dense(ni=rank, nj=rank)
            self.V = HD.Dense(ni=rank, nj=n)


        else:
            raise TypeError

    def __iadd__(self, A):
        assert self.dim[0] == A.dim[0] and self.dim[1] == A.dim[1]
        if self.rank + A.rank >= self.dim[0]:
            self = LowRank(HD.Dense(self) + HD.Dense(A), rank=self.rank)
        else:
            B = LowRank(
                m=self.dim[0], n=self.dim[1], rank=self.rank+A.rank,
                i_abs=self.i_abs, j_abs=self.j_abs, level=self.level
            )
            B.mergeU(self, A)
            B.mergeS(self, A)
            B.mergeV(self, A)
            self.rank += A.rank
            self.U = B.U
            self.S = B.S
            self.V = B.V

    def norm(self):
        return HD.Dense(self).norm()

    def mergeU(self, A, B):
        assert self.rank == A.rank + B.rank
        for i in range(self.dim[0]):
            for j in range(A.rank):
                self.U[i, j] = A.U[i, j]
            for j in range(B.rank):
                self.U[i, j+A.rank] = B.U[i, j]

    def mergeS(self, A, B):
        for i in range(A.rank):
            for j in range(A.rank):
                self.S[i, j] = A.S[i, j]
            for j in range(B.rank):
                self.S[i, j+A.rank] = 0
        for i in range(B.rank):
            for j in range(A.rank):
                self.S[i+A.rank, j] = 0
            for j in range(B.rank):
                self.S[i+A.rank, j+A.rank] = B.S[i, j]

    def mergeV(self, A, B):
        for i in range(A.rank):
            for j in range(self.dim[1]):
                self.V[i, j] = A.V[i, j]
        for i in range(B.rank):
            for j in range(self.dim[1]):
                self.V[i+A.rank, j] = B.V[i, j]

    def trsm(self, A, uplo):
        if isinstance(A, HD.Dense):
            if uplo == 'l':
                self.U.trsm(A, uplo)
            elif uplo == 'u':
                self.V.trsm(A, uplo)
        elif isinstance(A, HH.Hierarchical):
            if uplo == 'l':
                self.U.trsm(A, uplo)
            elif uplo == 'u':
                self.V.trsm(A, uplo)
        else:
            return NotImplemented

    def gemm(self, A, B, alpha=-1, beta=1):
        if isinstance(A, HD.Dense):
            if isinstance(B, LowRank):
                C = LowRank(B)
                C.U.gemm(A, B.U, alpha, 0)
                self += C
            else:
                return NotImplemented
        elif isinstance(A, LowRank):
            if isinstance(B, HD.Dense):
                C = LowRank(A)
                C.V.gemm(A.V, B, alpha, 0)
                self += C
            elif isinstance(B, LowRank):
                C = LowRank(A)
                C.V = HD.Dense(B.V)
                VxU = HD.Dense(ni=A.rank, nj=B.rank)
                VxU.gemm(A.V, B.U, 1, 0)
                SxVxU = HD.Dense(ni=A.rank, nj=B.rank)
                SxVxU.gemm(A.S, VxU, 1, 0)
                C.S.gemm(SxVxU, B.S, alpha, 0)
                self += C
            else:
                return NotImplemented
        else:
            return NotImplemented
