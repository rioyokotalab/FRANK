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
            A=None,
            func=None,
            ni=None,
            nj=None,
            i_begin=0,
            j_begin=0,
            i_abs=0,
            j_abs=0,
            level=0
    ):
        if isinstance(A, np.ndarray) and func is None:
            super().__init__(i_abs, j_abs, level)
            self.dim = [A.shape[0], A.shape[1]]
            self.data = A.copy()
        elif isinstance(A, np.ndarray) and func is not None:
            super().__init__(i_abs, j_abs, level)
            assert isinstance(ni, int) and isinstance(nj, int)
            assert callable(func)
            self.dim = [ni, nj]
            self.data = np.zeros((self.dim[0], self.dim[1]))
            func(self.data, A, ni, nj, i_begin, j_begin)
        elif isinstance(A, Dense):
            super().__init__(A.i_abs, A.j_abs, A.level)
            self.dim = A.dim
            self.data = A.data.copy()
        elif isinstance(A, HL.LowRank):
            super().__init__(A.i_abs, A.j_abs, A.level)
            self.dim = A.dim
            self.data = np.zeros((A.dim[0], A.dim[1]))
            UxS = Dense(ni=A.dim[0], nj=A.rank)
            UxS.gemm(A.U, A.S)
            self.gemm(UxS, A.V)
        elif isinstance(A, HH.Hierarchical):
            super().__init__(A.i_abs, A.j_abs, A.level)
            self.dim = [0, 0]
            for i in range(A.dim[0]):
                self.dim[0] += Dense(A[i, 0]).dim[0]
            for j in range(A.dim[1]):
                self.dim[1] += Dense(A[0, j]).dim[1]
            self.data = np.zeros((self.dim[0], self.dim[1]))
            i_begin = 0
            for i in range(A.dim[0]):
                AA = Dense(A[i, 0])
                j_begin = 0
                for j in range(A.dim[1]):
                    AD = Dense(A[i, j])
                    for ic in range(AD.dim[0]):
                        for jc in range(AD.dim[1]):
                            self[i_begin+ic, j_begin+jc] = AD[ic, jc]
                    j_begin += AD.dim[1]
                i_begin += AA.dim[0]
        elif A is None:
            super().__init__(i_abs, j_abs, level)
            assert isinstance(ni, int) or isinstance(nj, int)
            if ni is None:
                ni = 1
            elif nj is None:
                nj = 1
            self.dim = [ni, nj]
            self.data = np.zeros((ni, nj))
        else:
            raise ValueError

    def __add__(self, A):
        B = Dense(self)
        B += A
        return B

    def __sub__(self, A):
        B = Dense(self)
        B -= A
        return B

    def __iadd__(self, A):
        self.data += A.data
        return self

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
            self.data[pos] = data
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

    def resize(self, dim0, dim1):
        new_data = np.zeros((dim0, dim1))
        for i in range(dim0):
            for j in range(dim1):
                new_data[i, j] = self[i, j]
        self.dim = [dim0, dim1]
        self.data = new_data

    def getrf(self):
        x_getrf = sl.get_lapack_funcs('getrf', (self.data,))
        self.data, _, __ = x_getrf(self.data, overwrite_a=False)

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

    def gemm_trans(self, A, B, transA, transB, alpha, beta):
        x_gemm = sl.get_blas_funcs('gemm', (self.data, A.data, B.data))
        self.data = x_gemm(
            alpha, A.data, B.data, beta, self.data,
            overwrite_c=False, trans_a=transA, trans_b=transB)

    def gemm(self, A, B, alpha=-1, beta=1):
        if isinstance(A, Dense):
            if isinstance(B, Dense):
                assert A.dim[1] == B.dim[0]
                assert self.dim[1] == B.dim[1]
                if B.dim[1] == 1:
                    x_gemv = sl.get_blas_funcs(
                        'gemv', (self.data, A.data, B.data))
                    self.data = x_gemv(
                        alpha, A.data, B.data, beta, self.data,
                        overwrite_y=False, incx=1, incy=1, trans=False)
                else:
                    self.gemm_trans(A, B, False, False, alpha, beta)
            elif isinstance(B, HL.LowRank):
                AxU = Dense(ni=self.dim[0], nj=B.rank)
                AxU.gemm(A, B.U, 1, 0)
                AxUxS = Dense(ni=self.dim[0], nj=B.rank)
                AxUxS.gemm(AxU, B.S, 1, 0)
                self.gemm(AxUxS, B.V, alpha, beta)
            elif isinstance(B, HH.Hierarchical):
                C = HH.Hierarchical(self, ni_level=1, nj_level=B.dim[1])
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
                C = HH.Hierarchical(self, ni_level=1, nj_level=B.dim[1])
                C.gemm(A, B, alpha, beta)
                self.data = Dense(C).data
            else:
                return NotImplemented
        elif isinstance(A, HH.Hierarchical):
            if isinstance(B, Dense):
                C = HH.Hierarchical(self, ni_level=A.dim[0], nj_level=1)
                C.gemm(A, B, alpha, beta)
                self.data = Dense(C).data
            elif isinstance(B, HL.LowRank):
                C = HH.Hierarchical(self, ni_level=A.dim[0], nj_level=1)
                C.gemm(A, B, alpha, beta)
                self.data = Dense(C).data
            else:
                return NotImplemented
        else:
            return NotImplemented

    def qr(self, Q, R):
        for i in range(self.dim[1]):
            Q[i, i] = 1.0
        x_geqrf = sl.get_lapack_funcs('geqrf', (self.data,))
        self.data, tau, work, _ = x_geqrf(self.data, overwrite_a=False)
        x_ormqr = sl.get_lapack_funcs('ormqr', (self.data, tau, Q.data))
        Q.data, _, __ = x_ormqr(
            'L', 'N', self.data, tau, Q.data, lwork=len(work),
            overwrite_c=False)
        for i in range(self.dim[1]):
            for j in range(self.dim[1]):
                if j >= i:
                    R[i, j] = self[i, j]
        pass

    def svd(self, U, S, V):
        work = Dense(ni=self.dim[1]-1, nj=1)
        x_gesvd = sl.get_lapack_funcs('gesvd', (self.data,))
        U.data, Sdiag, V.data, _ = x_gesvd(
            self.data, overwrite_a=False, compute_uv=True, full_matrices=True)
        for i in range(self.dim[1]):
            S[i, i] = Sdiag[i]
