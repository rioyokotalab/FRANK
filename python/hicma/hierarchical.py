#!/bin/python
import numpy as np

from hicma.node import Node
from hicma.dense import Dense
from hicma.low_rank import LowRank


class Hierarchical(Node):
    """
    Recursive hirarchical matrix data structure.
    """
    def __init__(
            self,
            A=None,
            func=None,
            ni=None,
            nj=None,
            rank=None,
            nleaf=None,
            admis=1,
            ni_level=2,
            nj_level=2,
            i_begin=0,
            j_begin=0,
            i_abs=0,
            j_abs=0,
            level=0
    ):
        """
        Init the block matrix from the np.array A

        Arguments
        ---------
        self - np.array
        A - np.array or list
            Array or list from which this Hierarchical is initiated.
        func - function
            Function that is used to create Dense objects from vectors (green
            mesh etc).
        ni - int
            Number of rows of elements.
        nj - int
            Number of columns of elements.
        rank - int
            Target rank of admissible blocks.
        nleaf - int
            Maximum amount of rows or columns of a leaf node.
        admis - int
            Admissibility condition.
        i_begin - int
            Starting index in vector A for the rows.
        j_begin - int
            Starting index in vector A for the columns.
        ni_level - int
            Number of rows of blocks.
        nj_level - int
            Number of columns of blocks.
        level - int
            Level that the Hierarchical to be created resides at.
        """
        if isinstance(A, Dense):
            super().__init__(A.i_abs, A.j_abs, A.level)
            self.dim = [ni_level, nj_level]
            self.data = [None] * (self.dim[0] * self.dim[1])
            ni = A.dim[0]
            nj = A.dim[1]
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    ni_child = ni / self.dim[0]
                    if i == self.dim[0] - 1:
                        ni_child = ni - (ni / self.dim[0]) * (self.dim[0] - 1)
                    ni_child = int(ni_child)
                    nj_child = nj / self.dim[1]
                    if j == self.dim[1] - 1:
                        nj_child = nj - (nj / self.dim[1]) * (self.dim[1] - 1)
                    nj_child = int(nj_child)
                    i_abs_child = self.i_abs * self.dim[0] + i
                    j_abs_child = self.j_abs * self.dim[1] + j
                    self[i, j] = Dense(
                        ni=ni_child, nj=nj_child,
                        i_abs=i_abs_child, j_abs=j_abs_child, level=self.level + 1
                    )
                    i_begin = int(ni / self.dim[0] * i)
                    j_begin = int(nj / self.dim[1] * j)
                    for ic in range(ni_child):
                        for jc in range(nj_child):
                            self[i, j][ic, jc] = A[i_begin+ic, j_begin+jc]
        elif isinstance(A, LowRank):
            super().__init__(A.i_abs, A.j_abs, A.level)
            self.dim = [ni_level, nj_level]
            self.data = [None] * (self.dim[0] * self.dim[1])
            ni = A.dim[0]
            nj = A.dim[1]
            rank = A.rank
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    ni_child = ni / self.dim[0]
                    if i == self.dim[0] - 1:
                        ni_child = ni - (ni / self.dim[0]) * (self.dim[0] - 1)
                    ni_child = int(ni_child)
                    nj_child = nj / self.dim[1]
                    if j == self.dim[1] - 1:
                        nj_child = nj - (nj / self.dim[1]) * (self.dim[1] - 1)
                    nj_child = int(nj_child)
                    i_abs_child = self.i_abs * self.dim[0] + i
                    j_abs_child = self.j_abs * self.dim[1] + j
                    self[i, j] = LowRank(
                        m=ni_child, n=nj_child, k=rank,
                        i_abs=i_abs_child, j_abs=j_abs_child, level=self.level + 1
                    )
                    i_begin = int(ni / self.dim[0] * i)
                    j_begin = int(nj / self.dim[1] * j)
                    for ic in range(ni_child):
                        for kc in range(rank):
                            self[i, j].U[ic, kc] = A.U[i_begin+ic, kc]
                    self[i, j].S = A.S
                    for kc in range(rank):
                        for jc in range(nj_child):
                            self[i, j].V[kc, jc] = A.V[kc, j_begin+jc]
        elif isinstance(A, list):
            super().__init__(i_abs, j_abs, level)
            self.dim = [len(A), len(A[0])]
            self.data = [None] * (self.dim[0] * self.dim[1])
            for i in range(0, ni_level):
                for j in range(0, nj_level):
                    assert isinstance(A[i][j], Node)
                    self[i, j] = A[i][j]
                    self[i, j].i_abs = self.i_abs * self.dim[0] + i
                    self[i, j].j_abs = self.j_abs * self.dim[1] + j
                    self[i, j].level = self.level + 1
        elif isinstance(A, np.ndarray):
            super().__init__(i_abs, j_abs, level)
            self.dim = [min(ni_level, ni), min(nj_level, nj)]
            self.data = [None] * (self.dim[0] * self.dim[1])
            # Recursively create subblocks
            for i in range(0, self.dim[0]):
                for j in range(0, self.dim[1]):
                    ni_child = ni / self.dim[0];
                    if i == self.dim[0] - 1:
                        ni_child = ni - (ni/self.dim[0]) * (self.dim[0]-1);
                    ni_child = int(ni_child)
                    nj_child = nj / self.dim[1];
                    if j == self.dim[1] - 1:
                        nj_child = nj - (nj/self.dim[1]) * (self.dim[1]-1);
                    nj_child = int(nj_child)
                    i_begin_child = i_begin + i*int(ni / self.dim[0])
                    j_begin_child = j_begin + j*int(nj / self.dim[1])
                    i_abs_child = self.i_abs * self.dim[0] + i
                    j_abs_child = self.j_abs * self.dim[1] + j
                    if abs(i_abs_child - j_abs_child) <= admis\
                    or (ni == 1 or nj == 1):
                        if ni_child/ni_level < nleaf\
                        and nj_child/nj_level < nleaf:
                            self[i, j] = Dense(
                                A,
                                func,
                                ni_child,
                                nj_child,
                                i_begin_child,
                                j_begin_child,
                                i_abs_child,
                                j_abs_child,
                                level+1
                            )
                        else:
                            self[i, j] = Hierarchical(
                                A,
                                func,
                                ni_child,
                                nj_child,
                                rank,
                                nleaf,
                                admis,
                                ni_level,
                                nj_level,
                                i_begin_child,
                                j_begin_child,
                                i_abs_child,
                                j_abs_child,
                                level+1
                            )
                    else:
                        self[i, j] = LowRank(
                            Dense(
                                A,
                                func,
                                ni_child,
                                nj_child,
                                i_begin_child,
                                j_begin_child,
                                i_abs_child,
                                j_abs_child,
                                level+1
                            ),
                            k=rank
                        )
        else:
            raise TypeError

    def __getitem__(self, pos):
        """
        Get an item of the Hierarchical matrix from coordinates.

        Get one of the blocks held by the Hierarchical matrix. data is a
        one-dimensional array that can be indexed with just one coordinate, but
        represents a two-dimensional structure. This function allows accessing
        it with both one and two coordinates.

        Arguments
        ---------
        pos : tuple
            pos must have either one (i) or two (i, j) coordinates as elements.
        """
        if isinstance(pos, int):
            assert pos < self.dim[0] * self.dim[1]
            return self.data[pos]
        elif len(pos) == 2:
            i, j = pos
            assert i < self.dim[0] and j < self.dim[1]
            return self.data[i*self.dim[1] + j]
        else:
            raise ValueError

    def __setitem__(self, pos, data):
        """
        Recurse into the Hierarchical and set the proper submatrix/element.

        Set one of the blocks held by the Hierarchical matrix. data is a
        one-dimensional array that can be indexed with just one coordinate, but
        represents a two-dimensional structure. This function allows setting
        it with both one and two coordinates.

        Arguments
        ---------
        pos : tuple
            pos must have either one (i) or two (i, j) coordinates as elements.
        """
        if isinstance(pos, int):
            assert pos < self.dim[0] * self.dim[1]
            assert isinstance(data, Node)
            data.i_abs = self.i_abs * self.dim[0] + pos
            data.j_abs = self.j_abs * self.dim[1]
            data.level = self.level + 1
            self.data[pos] = data
        elif len(pos) == 2:
            i, j = pos
            assert i < self.dim[0] and j < self.dim[1]
            assert isinstance(data, Node)
            data.i_abs = self.i_abs * self.dim[0] + i
            data.j_abs = self.j_abs * self.dim[1] + j
            data.level = self.level + 1
            self.data[i * self.dim[1] + j] = data
        else:
            raise ValueError

    def norm(self):
        l2 = 0
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                l2 += self[i, j].norm()
        return l2

    def getrf(self):
        """
        Calculate the LU decomposition recursively

        Arguments
        ---------
        """
        for i in range(self.dim[0]):
            self[i, i].getrf()
            for j in range(i+1, self.dim[0]):
                self[i, j].trsm(self[i, i], 'l')
                self[j, i].trsm(self[i, i], 'u')
            for j in range(i+1, self.dim[0]):
                for k in range(i+1, self.dim[0]):
                    self[j, k].gemm(self[j, i], self[i, k], -1, 1)

    def trsm(self, A, uplo):
        assert isinstance(A, Hierarchical)
        if self.dim[1] == 1:
            if uplo == 'l':
                for i in range(self.dim[0]):
                    for j in range(i):
                        self[i].gemm(A[i, j], self[j], -1, 1)
                    self[i].trsm(A[i, i], 'l')
            elif uplo == 'u':
                for i in range(self.dim[0]-1, -1, -1):
                    for j in range(self.dim[0]-1, i, -1):
                        self[i].gemm(A[i, j], self[j], -1, 1)
                    self[i].trsm(A[i, i], 'u')
        else:
            if uplo == 'l':
                for j in range(self.dim[1]):
                    for i in range(self.dim[0]):
                        self.gemm_row(A, self, i, j, 0, i, -1, 1)
                        self[i, j].trsm(A[i, i], 'l')
            elif uplo == 'u':
                for i in range(self.dim[0]):
                    for j in range(self.dim[1]):
                        self.gemm_row(self, A, i, j, 0, j, -1, 1)
                        self[i, j].trsm(A[j, j], 'u')

    def gemm(self, A, B, alpha=-1, beta=1):
        if isinstance(A, Dense):
            if isinstance(B, Hierarchical):
                AH = Hierarchical(A, ni_level=self.dim[0], nj_level=B.dim[0])
                for i in range(self.dim[0]):
                    for j in range(self.dim[1]):
                        self.gemm_row(AH, B, i, j, 0, AH.dim[1], alpha, beta)
            else:
                raise NotImplemented
        elif isinstance(A, LowRank):
            if isinstance(B, LowRank):
                AH = Hierarchical(A, ni_level=self.dim[0], nj_level=self.dim[0])
                BH = Hierarchical(B, ni_level=self.dim[1], nj_level=self.dim[1])
                for i in range(self.dim[0]):
                    for j in range(self.dim[1]):
                        self.gemm_row(AH, BH, i, j, 0, AH.dim[1], alpha, beta)
            elif isinstance(B, Hierarchical):
                AH = Hierarchical(A, ni_level=self.dim[0], nj_level=B.dim[0])
                for i in range(self.dim[0]):
                    for j in range(self.dim[1]):
                        self.gemm_row(AH, B, i, j, 0, AH.dim[1], alpha, beta)
            else:
                raise NotImplemented
        elif isinstance(A, Hierarchical):
            if isinstance(B, Dense):
                BH = Hierarchical(B, ni_level=A.dim[1], nj_level=self.dim[1])
                for i in range(self.dim[0]):
                    for j in range(self.dim[1]):
                        self.gemm_row(A, BH, i, j, 0, A.dim[1], alpha, beta)
            elif isinstance(B, LowRank):
                BH = Hierarchical(B, ni_level=A.dim[1], nj_level=self.dim[1])
                for i in range(self.dim[0]):
                    for j in range(self.dim[1]):
                        self.gemm_row(A, BH, i, j, 0, A.dim[1], alpha, beta)
            elif isinstance(B, Hierarchical):
                assert self.dim[0] == A.dim[0] and self.dim[1] == B.dim[1]
                assert A.dim[1] == B.dim[0]
                for i in range(self.dim[0]):
                    for j in range(self.dim[1]):
                        self.gemm_row(A, B, i, j, 0, A.dim[1], alpha, beta)
            else:
                raise NotImplemented
        else:
            raise NotImplemented

    def gemm_row(self, A, B, i, j, k_min, k_max, alpha, beta):
        rank = -1
        if isinstance(self[i, j], LowRank):
            rank = self[i, j].rank
            self[i, j] = Dense(self[i, j])
        for k in range(k_min, k_max):
            self[i, j].gemm(A[i, k], B[k, j], alpha, beta)
        if rank != -1:
            assert isinstance(self[i, j], Dense)
            self[i, j] = LowRank(self[i, j], k=rank)
