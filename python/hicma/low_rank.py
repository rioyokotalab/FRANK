#!/bin/python
import random
import numpy as np

from hicma.node import Node
import hicma.dense as HD
import hicma.hierarchical as HH


class LowRank(Node):
    def __init__(
            self,
            A=None,
            m=None,
            n=None,
            k=None,
            i_abs=0,
            j_abs=0,
            level=0
    ):
        if isinstance(A, HD.Dense):
            super().__init__(A.i_abs, A.j_abs, A.level)
            self.dim = A.dim
            assert isinstance(k, int)
            self.rank = min(min(k+5, self.dim[0]), self.dim[1])
            self.U = HD.Dense(
                ni=self.dim[0], nj=k, i_abs=i_abs, j_abs=j_abs, level=level)
            self.S = HD.Dense(
                ni=self.rank, nj=self.rank, i_abs=i_abs, j_abs=j_abs, level=level)
            self.V = HD.Dense(
                ni=k, nj=self.dim[1], i_abs=i_abs, j_abs=j_abs, level=level)
            RN = HD.Dense(
                np.random.normal(0, 1, self.dim[1]*self.rank).reshape(
                    self.dim[1], self.rank))
            Y = HD.Dense(ni=self.dim[0], nj=self.rank)
            Y.gemm_trans(A, RN, False, False, 1, 0) # Y = A *  RN
            Q = HD.Dense(ni=self.dim[0], nj=self.rank)
            R = HD.Dense(ni=self.rank, nj=self.rank)
            Y.qr(Q, R) # [Q, R] = qr(Y)
            Bt = HD.Dense(ni=self.dim[1], nj=self.rank)
            Bt.gemm_trans(A, Q, True, False, 1, 0) # B' = A' * Q
            Qb = HD.Dense(ni=self.dim[1], nj=self.rank)
            Rb = HD.Dense(ni=self.rank, nj=self.rank)
            Bt.qr(Qb, Rb) # [Qb, Rb] = qr(B')
            Ur = HD.Dense(ni=self.rank, nj=self.rank)
            Vr = HD.Dense(ni=self.rank, nj=self.rank)
            Rb.svd(Vr, self.S, Ur) # [Vr, S, Ur] = svd(Rb)
            Ur.resize(k, self.rank)
            self.U.gemm_trans(Q, Ur, False, True, 1, 0) # U = Q * Ur'
            Vr.resize(self.rank, k)
            self.V.gemm_trans(Vr, Qb, True, True, 1, 0) # V = Vr' * Qb'
            self.S.resize(k, k)
            self.rank = k
        elif isinstance(A, LowRank):
            super().__init__(A.i_abs, A.j_abs, A.level)
            self.dim = A.dim
            self.rank = A.rank
            self.U = HD.Dense(A.U)
            self.S = HD.Dense(A.S)
            self.V = HD.Dense(A.V)
        elif A is None:
            super().__init__(i_abs, j_abs, level)
            assert isinstance(m, int) and isinstance(n, int)
            assert isinstance(k, int)
            self.dim = [m, n]
            self.rank = k
            self.U = HD.Dense(ni=m, nj=k, i_abs=i_abs, j_abs=j_abs, level=level)
            self.S = HD.Dense(ni=k, nj=k, i_abs=i_abs, j_abs=j_abs, level=level)
            self.V = HD.Dense(ni=k, nj=n, i_abs=i_abs, j_abs=j_abs, level=level)
        else:
            raise TypeError

    def __iadd__(self, A):
        assert self.dim[0] == A.dim[0] and self.dim[1] == A.dim[1]
        if self.rank + A.rank >= self.dim[0]:
            self = LowRank(HD.Dense(self) + HD.Dense(A), k=self.rank)
        else:
            B = LowRank(
                m=self.dim[0], n=self.dim[1], k=self.rank+A.rank,
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
        assert self.rank == A.rank + B.rank
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
        assert self.rank == A.rank + B.rank
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
                H = HH.Hierarchical(self.U, ni_level=A.dim[0], nj_level=1)
                H.trsm(A, uplo)
                self.U = HD.Dense(H)
            elif uplo == 'u':
                H = HH.Hierarchical(self.V, ni_level=1, nj_level=A.dim[1])
                H.trsm(A, uplo)
                self.V = HD.Dense(H)
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
