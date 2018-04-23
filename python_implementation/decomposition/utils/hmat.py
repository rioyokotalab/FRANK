#!/bin/python
import numpy as np

from decomposition.utils.node import Node
from decomposition.utils.emat import EMat
from decomposition.utils.dmat import DMat
from decomposition.utils.lrmat import LRMat


class HMat(Node):
    """
    Recursive hirarchical matrix data structure.
    """
    def __init__(
            self,
            func=None,
            arr=None,
            parent=None,
            k=None,
            p=None,
            max_n_leaf=None,
            admis=0,
            ni_level=2,
            nj_level=2,
            pos=None
    ):
        """
        Init the block matrix from the np.array arr

        Arguments
        ---------
        self - np.array
        arr - np.array or list
            Array or list from which this HMat is initiated.
        parent - HMat
            Parent of the HMat to be created.
        pos - tuple
            Holding the row and column index that this object should have
            within its parent.
        max_n_leaf - int
            Maximum amount of elements for a leaf node. In matrix form, this
            corresponds to the number of rows and columns (which are equal)
        k - int
            Target rank of admissible blocks.
        p - int
            Overscan during low rank matrix creation.
        """
        super().__init__(parent, pos)
        self._init_position(pos)
        if parent is not None:
            self.max_n_leaf = parent.max_n_leaf
            self.k = parent.k
            self.p = parent.p
        else:
            # assert max_n_leaf is not None
            # assert k is not None
            # assert p is not None
            self.max_n_leaf = max_n_leaf
            self.k = k
            self.p = p
        self.blocks = [
            [None, None],
            [None, None]
        ]
        self._init_substructure(arr, admis, ni_level, nj_level)

    def _init_substructure(self, arr, admis, ni_level, nj_level):
        """
        Initiate the substructure from the given input data in arr.

        Arguments
        ---------
        arr - np.array or list
            Array or list from which this HMat is initiated.
        """
        if isinstance(arr, list):
            assert len(arr) == ni_level
            assert len(arr[0]) == nj_level
            for i in range(0, ni_level):
                for j in range(0, nj_level):
                    assert isinstance(arr[i][j], Node)
                    arr[i][j].update_position(self, (i, j))
                    self[i, j] = arr[i][j]

        elif isinstance(arr, np.ndarray):
            # Recursively create subblocks
            ni_subarr = int(arr.shape[0] / ni_level)
            nj_subarr = int(arr.shape[1] / nj_level)
            for i in range(0, ni_level):
                for j in range(0, nj_level):
                    row = self.row * ni_level + i
                    col = self.col * nj_level + j
                    if i == ni_level - 1:
                        i_max = arr.shape[0]
                    else:
                        i_max = (i + 1) * ni_subarr
                    if j == nj_level - 1:
                        j_max = arr.shape[1]
                    else:
                        j_max = (j + 1) * nj_subarr
                    subarr = arr[
                        i*ni_subarr:i_max,
                        j*nj_subarr:j_max
                    ]
                    if abs(row - col) <= admis:
                        if min(ni_subarr, nj_subarr) <= self.max_n_leaf:
                            self[i, j] = DMat(
                                subarr,
                                self,
                                (i, j)
                            )
                        else:
                            self[i, j] = HMat(
                                arr=subarr,
                                parent=self,
                                pos=(i, j),
                            )
                    else:
                        self[i, j] = LRMat(
                            subarr,
                            self.k,
                            self.p,
                            self,
                            (i, j)
                        )
        else:
            raise TypeError

    def update_position(self, parent, pos):
        """
        Recursively update the position data of the HMat.
        """
        self.parent = parent
        self.root = self.parent is None
        self._init_position(pos)
        for i in range(0, 2):
            for j in range(0, 2):
                self.blocks[i][j].update_position(self, (i, j))

    def get_dense(self):
        return np.block([
            [
                self.blocks[i][j].get_dense()
                for j in range(0, 2)
            ]
            for i in range(0, 2)
        ])

    def _get_lower_triangle(self):
        """
        Recursively recovers the lower triangle.
        """
        return HMat(
            arr=[
                [
                    self[0, 0]._get_lower_triangle(),
                    EMat()
                ],
                [
                    self[1, 0],
                    self[1, 1]._get_lower_triangle()
                ]
            ],
            max_n_leaf=self.max_n_leaf,
            k=self.k,
            p=self.p
        )

    def _get_upper_triangle(self):
        """
        Recursively recovers the upper triangle.
        """
        return HMat(
            arr=[
                [
                    self[0, 0]._get_upper_triangle(),
                    self[0, 1]
                ],
                [

                    EMat(),
                    self[1, 1]._get_upper_triangle()
                ]
            ],
            max_n_leaf=self.max_n_leaf,
            k=self.k,
            p=self.p
        )

    def reconstruct_from_lu(self):
        """
        Efficiently reconstruct the matrix from the LU decomposition.

        This function should only be called to verify the result after a doing
        a LU decomposition of the HMat.
        """
        L = self._get_lower_triangle()
        U = self._get_upper_triangle()
        if isinstance(L, DMat) and isinstance(U, DMat):
            return (L @ U).get_dense()
        elif isinstance(L, HMat) and isinstance(U, HMat):
            # NOTE Not working because off_diag data is used by accident.
            # Find way to add zero matrix for off diagonal without doing
            # additional calculation
            return np.block(
                [
                    [
                        (L[0, 0] @ U[0, 0]).get_dense(),
                        (L[0, 0] @ U[0, 1]).get_dense()
                    ],
                    [
                        (L[1, 0] @ U[0, 0]).get_dense(),
                        (L[1, 0] @ U[0, 1] + L[1, 1] @ U[1, 1]).get_dense()
                    ]
                ]
            )
        else:
            raise ValueError

    def lower_trsm(self, other):
        """
        Multiply other with invert of lower triagonal of self.

        Uses the following formula
        inv( |A  | ) = |         inv(A)           0   |
           ( |C D| )   |-(inv(D) @ B @ inv(A))  inv(D)|
        inv(A) and inv(D) are computed in recursion. The lowest level is
        handled in the DMat class and uses the level 3 BLAS function trsm.
        On the HMat level, only multiplications and additions are called.

        Arguments
        ---------
        other - h_mat or lr_mat
            Target of the trsm operation.
        """
        # inv_lt(HMat) @ LRMat
        if isinstance(other, LRMat):
            # a-upper: inv(A) @ upper_half(U)
            au = self[0, 0].lower_trsm(other[0]._upper()).get_dense()
            # d-lower: inv(D) @ lower_half(U)
            dl = self[1, 1].lower_trsm(other[0]._lower()).get_dense()
            # inv(D) @ C
            dc = self[1, 1].lower_trsm(self[1, 0]).get_dense()
            return LRMat(
                [
                    DMat(
                        np.vstack((
                            au,
                            - dc @ au + dl
                        ))
                    ),
                    other[1],
                    other[2]
                ],
                k=other.k,
                p=other.p
            )
        # inv_lt(HMat) @ DMat
        elif isinstance(other, DMat):
            # a-upper: inv(A) @ upper_half(U)
            au = self[0, 0].lower_trsm(other._upper()).get_dense()
            # d-lower: inv(D) @ lower_half(U)
            dl = self[1, 1].lower_trsm(other._lower()).get_dense()
            # inv(D) @ C
            dc = self[1, 1].lower_trsm(self[1, 0]).get_dense()
            return DMat(
                np.vstack((
                    au,
                    - dc @ au + dl
                ))
            )
        # inv_lt(HMat) @ HMat
        elif isinstance(other, HMat):
            # a-top-left: inv(A) @ top_left(HMat)
            atl = self[0, 0].lower_trsm(other[0, 0])
            # a-top-right: inv(A) @ top_right(HMat)
            atr = self[0, 0].lower_trsm(other[0, 1])
            # d-bottom-left: inv(D) @ bottom_left(HMat)
            dbl = self[1, 1].lower_trsm(other[1, 0])
            # d-bottom-right: inv(D) @ bottom_right(HMat)
            dbr = self[1, 1].lower_trsm(other[1, 1])
            # inv(D) @ C
            dc = self[1, 1].lower_trsm(self[1, 0])

            return HMat(
                arr=[
                    [
                        atl,
                        atr
                    ],
                    [
                        - dc @ atl + dbl,
                        - dc @ atr + dbr
                    ]
                ],
                parent=self.parent,
                pos=self.pos
            )
        else:
            raise ValueError

    def upper_trsm(self, other):
        """
        Multiply other with invert of upper triagonal of self.

        Uses the following formula
        inv( |A B| ) = |inv(A)  -(inv(A) @ B inv(D)) |
           ( |0 D| )   |  0             inv(D)       |
        inv(A) and inv(D) are computed in recursion. The lowest level is
        handled in the DMat class and uses the level 3 BLAS function trsm.
        On the HMat level, only multiplications and additions are called.

        Arguments
        ---------
        other - h_mat or lr_mat
            Target of the trsm operation.
        """
        # LRMat @ inv_ut(HMat)
        if isinstance(other, LRMat):
            # left-a: left_half(V) @ inv(A)
            la = self[0, 0].upper_trsm(other[2]._left()).get_dense()
            # right-d: right_half(V) @ inv(D)
            rd = self[1, 1].upper_trsm(other[2]._right()).get_dense()
            # B @ inv(D)
            bd = self[1, 1].upper_trsm(self[0, 1]).get_dense()
            return LRMat(
                [
                    other[0],
                    other[1],
                    DMat(
                        np.hstack((
                            la,
                            - la @ bd + rd
                        ))
                    )
                ],
                k=other.k,
                p=other.p
            )
        # DMat @ inv_ut(HMat)
        elif isinstance(other, DMat):
            # left-a: left_half(V) @ inv(A)
            la = self[0, 0].upper_trsm(other._left()).get_dense()
            # right-d: right_half(V) @ inv(D)
            rd = self[1, 1].upper_trsm(other._right()).get_dense()
            # B @ inv(D)
            bd = self[1, 1].upper_trsm(self[0, 1]).get_dense()
            return DMat(
                np.hstack((
                    la,
                    - la @ bd + rd
                )),
            )
        # HMat @ inv_ut(HMat)
        if isinstance(other, HMat):
            # top-left-a: top_left(HMat) @ inv(A)
            tla = self[0, 0].upper_trsm(other[0, 0])
            # bottom-left-a: @ bottom_left(HMat) @ inv(A)
            bla = self[0, 0].upper_trsm(other[1, 0])
            # top-right-d: top_right(HMat) @ inv(D)
            trd = self[1, 1].upper_trsm(other[0, 1])
            # bottom-right-d: bottom_right(HMat) @ inv(D)
            brd = self[1, 1].upper_trsm(other[1, 1])
            # B @ inv(D)
            bd = self[1, 1].upper_trsm(self[0, 1])

            return HMat(
                arr=[
                    [
                        tla,
                        - tla @ bd + trd
                    ],
                    [
                        bla,
                        - bla @ bd + brd
                    ]
                ],
                parent=self.parent,
                pos=self.pos
            )
        else:
            raise ValueError

    def __getitem__(self, pos):
        """
        Recuse into the HMat and return the proper submatrix/element.

        pos is a tuple of either lenth two or three (see arguments section).
        If the level is specified, the desired block is recursively found.
        Additionally, if the level is specified, row and block must both be
        smaller than 2**level.
        If level is smaller than 1,
        row and col are transformed into a Morton bitstring. The shorter of the
        bitstrings for col and row is padded with zeros on the left to be of
        the same length. The first bit of the respective strings is removed and
        used to determine the subblock of the matrix to recurse into/return.
        If the string is not empty at the lowest level, a single matrix of the
        current block is returned according to the leftover strings.
        Example:
            row = 4 <=> bin(row) = '0b100'
            col = 1 <=> bin(col) = '0b1'
            Padding: row = '0b001'
            Pick block [1, 0] and recurse with row = '0b00' and col = '0b00'

        Arguments
        ---------
        pos : tuple
            pos must have either two or three elements. In the case of two
            elements, it is HMat[row, col]. In the case of three elements, it
            is HMat[level, row, col]
        """
        if len(pos) == 2:
            row, col = pos
            return self.blocks[row][col]
        else:
            level, row, col = pos
            # Calculate proper length Morton indices
            row_b = bin(row)[2:]
            row_b = '0' * (level - len(row_b) + 1) + row_b
            col_b = bin(col)[2:]
            col_b = '0' * (level - len(col_b) + 1) + col_b
            # If end of morton indexing is reached
            if len(row_b) == 1 and len(col_b) == 1:
                return self.blocks[row][col]
            # Otherwise recurse using the the rest of the bitstrings
            else:
                assert isinstance(
                    self.blocks[int(row_b[0], 2)][int(col_b[0], 2)],
                    HMat
                ), "Depth of matrix hierarchy exceeded!"
                return self.blocks[int(row_b[0], 2)][int(col_b[0], 2)][
                    level-1, int(row_b[1:], 2), int(col_b[1:], 2)
                ]

    def __setitem__(self, pos, data):
        """
        Recurse into the HMat and set the proper submatrix/element.

        pos is a tuple of either lenth two or three (see arguments section).
        If the level is specified, the desired block is recursively found.
        Additionally, if the level is specified, row and block must both be
        smaller than 2**level.
        If level is smaller than 1,
        row and col are transformed into a Morton bitstring. The shorter of the
        bitstrings for col and row is padded with zeros on the left to be of
        the same length. The first bit of the respective strings is removed and
        used to determine the subblock of the matrix to recurse into/return.
        If the string is not empty at the lowest level, a single matrix of the
        current block is returned according to the leftover strings.
        empty a
        Example:
            row = 4 <=> bin(row) = '0b100'
            col = 1 <=> bin(col) = '0b1'
            Padding: row = '0b001'
            Pick block [1, 0] and recurse with row = '0b00' and col = '0b00'

        Arguments
        ---------
        pos : tuple
            pos must have either two or three elements. In the case of two
            elements, it is HMat[row, col]. In the case of three elements, it
            is HMat[level, row, col]
        """
        if len(pos) == 2:
            row, col = pos
            if isinstance(data, np.ndarray):
                assert isinstance(self.blocks[row][col], DMat)
                self.blocks[row][col].arr = data
            elif isinstance(data, Node):
                data.update_position(self, (row, col))
                self.blocks[row][col] = data
            else:
                raise ValueError('Bad data for setting item!')
        else:
            level, row, col = pos
            # Otherwise use morton indexing
            row_b = bin(row)[2:]
            row_b = '0' * (level - len(row_b) + 1) + row_b
            col_b = bin(col)[2:]
            col_b = '0' * (level - len(col_b) + 1) + col_b
            # If end of morton indexing is reached
            if len(row_b) == 1 and len(col_b) == 1:
                self[row, col] = data
            else:
                # And recurse using the the rest of the bitstrings
                assert isinstance(
                    self.blocks[int(row_b[0], 2)][int(col_b[0], 2)],
                    HMat
                ), "Depth of matrix hierarchy exceeded!"
                self.blocks[int(row_b[0], 2)][int(col_b[0], 2)][
                    level-1, int(row_b[1:], 2), int(col_b[1:], 2)
                ] = data

    def __matmul__(self, other):
        # HMat @ D_MAT
        # Note that this only happens when DMat is used as U in a
        # UBV decomposition
        if isinstance(other, DMat):
            return DMat(
                np.vstack((
                    (
                        self[0, 0] @ other._upper()
                        + self[0, 1] @ other._lower()
                    ).arr,
                    (
                        self[1, 0] @ other._upper()
                        + self[1, 1] @ other._lower()
                    ).arr
                ))
            )
        # HMat @ LRMat
        elif isinstance(other, LRMat):
            return LRMat(
                [
                    self @ other[0],
                    other[1],
                    other[2]
                ],
                k=other.k,
                p=other.p
            )
        # HMat @ HMat
        elif isinstance(other, HMat):
            return HMat(
                arr=[
                    [
                        self[0, 0] @ other[0, 0] + self[0, 1] @ other[1, 0],
                        self[0, 0] @ other[0, 1] + self[0, 1] @ other[1, 1]
                    ],
                    [
                        self[1, 0] @ other[0, 0] + self[1, 1] @ other[1, 0],
                        self[1, 0] @ other[0, 1] + self[1, 1] @ other[1, 1]
                    ]
                ],
                parent=self.parent,
                pos=self.pos
            )
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        # DMat @ HMat
        # Note that this only happens when DMat is used as V in a
        # UBV decomposition
        if isinstance(other, DMat):
            return DMat(
                np.hstack((
                    (
                        other._left() @ self[0, 0]
                        + other._right() @ self[1, 0]
                    ).arr,
                    (
                        other._left() @ self[0, 1]
                        + other._right() @ self[1, 1]
                    ).arr
                ))
            )
        # LRMat @ HMat
        elif isinstance(other, LRMat):
            return LRMat(
                [
                    other[0],
                    other[1],
                    other[2] @ self
                ],
                k=other.k,
                p=other.p
            )
        else:
            return NotImplemented

    def __add__(self, other):
        # HMat + HMat
        if isinstance(other, HMat):
            return HMat(
                arr=[
                    [
                        self[0, 0] + other[0, 0],
                        self[0, 1] + other[0, 1]
                    ],
                    [
                        self[1, 0] + other[1, 0],
                        self[1, 1] + other[1, 1]
                    ]
                ],
                parent=self.parent,
                pos=self.pos
            )
        else:
            return NotImplemented

    def __radd__(self, other):
        # LRMat + HMat
        if isinstance(other, LRMat):
            return HMat(
                arr=other.get_dense() + self.get_dense(),
                parent=self.parent,
                pos=self.pos
            )
        else:
            return NotImplemented

    def __sub__(self, other):
        # HMat - HMat
        if isinstance(other, HMat):
            return HMat(
                arr=[
                    [self[0, 0] - other[0, 0], self[0, 1] - other[0, 1]],
                    [self[1, 0] - other[1, 0], self[1, 1] - other[1, 1]]
                ],
                parent=self.parent,
                pos=self.pos
            )
        # HMat - LRMat
        elif isinstance(other, LRMat):
            return HMat(
                arr=self.get_dense() - other.get_dense(),
                parent=self.parent,
                pos=self.pos
            )
        else:
            return NotImplemented

    def __isub__(self, other):
        # HMat -= LRMat
        if isinstance(other, LRMat):
            return HMat(
                arr=self.get_dense() - other.get_dense(),
                parent=self.parent,
                pos=self.pos
            )
        else:
            return NotImplemented

    def __neg__(self):
        return HMat(
            arr=[
                [-self[0, 0], -self[0, 1]],
                [-self[1, 0], -self[1, 1]]
            ],
            parent=self.parent,
            pos=self.pos
        )
