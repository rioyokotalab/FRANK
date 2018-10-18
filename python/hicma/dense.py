#!/bin/python
import numpy as np
import numbers
import scipy.linalg as sl

import hicma.low_rank as HL
import hicma.hierarchical as HH
from hicma.node import Node
from hicma.id import lup_factorize


class Dense(Node):
    """
    Dense matrix container.
    """
    # TODO Pass generator function
    def __init__(
            self,
            arr=None,
            func=None,
            ni=None,
            nj=None,
            i_begin=0,
            j_begin=0,
            i_abs=0,
            j_abs=0,
            level=0
    ):
        if isinstance(arr, np.ndarray) and func is None:
            super().__init__(i_abs, j_abs, level)
            self.dim = [arr.shape[0], arr.shape[1]]
            self.data = arr.copy()
        elif isinstance(arr, np.ndarray) and func is not None:
            super().__init__(i_abs, j_abs, level)
            assert isinstance(ni, int) and isinstance(nj, int)
            self.dim = [ni, nj]
            self.data = np.zeros((self.dim[0], self.dim[1]))
            func(self.data, arr, ni, nj, i_begin, j_begin)
        elif isinstance(arr, Dense):
            super().__init__(arr.i_abs, arr.j_abs, arr.level)
            self.dim = arr.dim
            self.data = arr.data.copy()
        elif isinstance(arr, HL.LowRank):
            super().__init__(arr.i_abs, arr.j_abs, arr.level)
            self.dim = arr.dim
            UxS = Dense(ni=arr.dim[0], nj=arr.rank)
            UxS.gemm(arr.U, arr.S)
            self.data = np.zeros((self.dim[0], self.dim[1]))
            self.gemm(UxS, arr.V)
        elif isinstance(arr, HH.Hierarchical):
            super().__init__(arr.i_abs, arr.j_abs, arr.level)
            self.dim = [0, 0]
            for i in range(arr.dim[0]):
                self.dim[0] += Dense(arr[i, 0]).dim[0]
            for j in range(arr.dim[1]):
                self.dim[1] += Dense(arr[0, j]).dim[1]
            self.data = np.zeros((self.dim[0], self.dim[1]))
            i_begin = 0
            for i in range(arr.dim[0]):
                AA = Dense(arr[i, 0])
                j_begin = 0
                for j in range(arr.dim[1]):
                    AD = Dense(arr[i, j])
                    for ic in range(AD.dim[0]):
                        for jc in range(AD.dim[1]):
                            self[i_begin+ic, j_begin+jc] = AD[ic, jc]
                    j_begin += AD.dim[1]
                i_begin += AA.dim[0]
        elif arr is None:
            super().__init__(i_abs, j_abs, level)
            assert isinstance(ni, int) and isinstance(nj, int)
            self.dim = [ni, nj]
            self.data = np.zeros((ni, nj))
        else:
            raise ValueError

    def __add__(self, A):
        return Dense(
            self.data + A.data,
            i_abs=self.i_abs, j_abs=self.j_abs, level=self.level)

    def __iadd__(self, A):
        self.data += A.data
        return self


    def __sub__(self, A):
        return Dense(
            self.data - A.data,
            i_abs=self.i_abs, j_abs=self.j_abs, level=self.level)

    def __isub__(self, A):
        self.data -= A.data
        return self

    def __getitem__(self, pos):
        if isinstance(pos, int):
            assert pos < self.dim[0] * self.dim[1]
            return self.data[pos]
        elif len(pos) == 2:
            i, j = pos
            assert i < self.dim[0] and j < self.dim[1]
            return self.data[i, j]
        else:
            raise ValueError


    def __setitem__(self, pos, data):
        if isinstance(pos, int):
            assert pos < self.dim[0] * self.dim[1]
            assert isinstance(data, numbers.Number)
            self.data[pos / self.dim[0], pos % self.dim[0]] = data
        elif len(pos) == 2:
            i, j = pos
            assert i < self.dim[0] and j < self.dim[1]
            assert isinstance(data, numbers.Number)
            self.data[i, j] = data
        else:
            raise ValueError

    def norm(self):
        l2 = 0
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                l2 += self[i, j] * self[i, j]
        return l2

    def getrf(self):
        self.data, piv = sl.lu_factor(self.data, False, True)
        for i, p in enumerate(piv):
            self.data[i], self.data[p] = self.data[p], self.data[i]

    def trsm(self, A, uplo):
        if isinstance(A, Dense):
            if self.dim[1] == 1:
                if uplo == 'l':
                    x_trsm = sl.get_blas_funcs('trsm', (A.data, self.data))
                    self.data = x_trsm(1, A.data, self.data, side=0,
                          lower=True, trans_a=False, diag=True)
                elif uplo == 'u':
                    x_trsm = sl.get_blas_funcs('trsm', (A.data, self.data))
                    self.data = x_trsm(1, A.data, self.data, side=0,
                          lower=False, trans_a=False, diag=False)
            else:
                if uplo == 'l':
                    x_trsm = sl.get_blas_funcs('trsm', (A.data, self.data))
                    self.data = x_trsm(1, A.data, self.data, side=0,
                          lower=True, trans_a=False, diag=True)
                elif uplo == 'u':
                    x_trsm = sl.get_blas_funcs('trsm', (A.data, self.data))
                    self.data = x_trsm(1, A.data, self.data, side=1,
                          lower=False, trans_a=False, diag=False)
        elif isinstance(A, HH.Hierarchical):
            if uplo == 'l':
                H = HH.Hierarchical(self, ni_level=A.dim[0], nj_level=1)
                H.trsm(A, uplo)
                self.data = Dense(H).data
            elif uplo == 'u':
                H = HH.Hierarchical(self, ni_level=1, nj_level=A.dim[1])
                H.trsm(A, uplo)
                self.data = Dense(H).data
        else:
            return NotImplemented

    def gemm(self, A, B, alpha=-1, beta=1):
        if isinstance(A, Dense):
            if isinstance(B, Dense):
                self.data = alpha*(A.data @ B.data) + beta * self.data
            elif isinstance(B, HL.LowRank):
                AxU = Dense(ni=self.dim[0], nj=B.rank)
                AxU.gemm(A, B.U, 1, 0)
                AxUxS = Dense(ni=self.dim[0], nj=B.rank)
                AxUxS.gemm(AxU, B.S, 1, 0)
                self.gemm(AxUxS, B.V, alpha, beta)
            elif isinstance(B, HH.Hierarchical):
                C = HH.Hierarchical(self, ni_level=B.dim[0], nj_level=B.dim[1])
                C.gemm(A, B, alpha, beta)
                self.data = Dense(C).data
            else:
                return NotImplemented
        elif isinstance(A, HL.LowRank):
            if isinstance(B, Dense):
                VxB = Dense(ni=A.rank, nj=self.dim[1])
                VxB.gemm(A.V, B, 1, 0)
                SxVxB = Dense(ni=A.rank, nj=self.dim[1])
                SxVxB.gemm(A.S, VxB, 1, 0)
                self.gemm(A.U, SxVxB, alpha, beta)
            elif isinstance(B, HL.LowRank):
                VxU = Dense(ni=A.rank, nj=B.rank)
                VxU.gemm(A.V, B.U, 1, 0)
                SxVxU = Dense(ni=A.rank, nj=B.rank)
                SxVxU.gemm(A.S, VxU, 1, 0)
                SxVxUxS = Dense(ni=A.rank, nj=B.rank)
                SxVxUxS.gemm(SxVxU, B.S, 1, 0)
                UxSxVxUxS = Dense(ni=A.dim[0], nj=B.rank)
                UxSxVxUxS.gemm(A.U, SxVxUxS, 1, 0)
                self.gemm(UxSxVxUxS, B.V, alpha, beta)
            elif isinstance(B, HH.Hierarchical):
                C = HH.Hierarchical(self, ni_level=B.dim[0], nj_level=B.dim[1])
                C.gemm(A, B, alpha, beta)
                self.data = Dense(C).data
            else:
                return NotImplemented
        elif isinstance(A, HH.Hierarchical):
            if isinstance(B, Dense):
                C = HH.Hierarchical(self, ni_level=A.dim[0], nj_level=A.dim[1])
                C.gemm(A, B, alpha, beta)
                self.data = Dense(C).data
            elif isinstance(B, HL.LowRank):
                C = HH.Hierarchical(self, ni_level=A.dim[0], nj_level=A.dim[1])
                C.gemm(A, B, alpha, beta)
                self.data = Dense(C).data
            else:
                return NotImplemented
        else:
            return NotImplemented
