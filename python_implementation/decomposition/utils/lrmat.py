#!/bin/python
import numpy as np

from decomposition.svd import rsvd
from decomposition.utils.node import Node
from decomposition.utils.dmat import DMat


class LRMat(Node):
    """
    Class for low-rank-form matrices.
    """
    def __init__(
            self,
            arr,
            k=None,
            p=None,
            parent=None,
            pos=None
    ):
        """
        Initialize from input data
        """
        self.k = k
        self.p = p
        super().__init__(parent, pos)
        if isinstance(arr, np.ndarray):
            assert k is not None
            assert p is not None
            U, D, V = rsvd(arr, k, p)
            self.blocks = [None, None, None]
            self.blocks[0] = DMat(U)
            self.blocks[1] = DMat(D)
            self.blocks[2] = DMat(V)
        elif isinstance(arr, list):
            assert len(arr) == 3
            assert isinstance(arr[0], DMat)
            assert isinstance(arr[1], DMat)
            assert isinstance(arr[2], DMat)
            assert arr[0].arr.shape[1] == arr[1].arr.shape[0]
            assert arr[1].arr.shape[1] == arr[2].arr.shape[0]

            self.blocks = arr
        else:
            raise TypeError

    def get_dense(self):
        return (self[0] @ self[1] @ self[2]).arr

    def __getitem__(self, pos):
        assert isinstance(pos, int), 'LRMat only has one dimension!'
        assert pos in [0, 1, 2]
        return self.blocks[pos]

    def __matmul__(self, other):
        # LRMat @ DMat
        if isinstance(other, DMat):
            return LRMat(
                [
                    self[0],
                    self[1],
                    self[2] @ other
                ],
                k=self.k,
                p=self.p
            )
        # LRMat @ LRMat
        elif isinstance(other, LRMat):
            return LRMat(
                [
                    self[0],
                    self[1] @ self[2] @ other[0] @ other[1],
                    other[2]
                ],
                k=self.k,
                p=self.p
            )
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        # DMat @ LRMat
        if isinstance(other, DMat):
            return LRMat(
                [
                    other @ self[0],
                    self[1],
                    self[2]
                ],
                k=self.k,
                p=self.p
            )
        else:
            return NotImplemented

    def __add__(self, other):
        # LRMat + DMat
        if isinstance(other, DMat):
            return DMat(
                self.get_dense() + other.arr
            )
        # LRMat + LRMat
        elif isinstance(other, LRMat):
            return LRMat(
                self.get_dense() + other.get_dense(),
                k=self.k,
                p=self.p
            )
        else:
            return NotImplemented

    def __radd__(self, other):
        # DMat + LRMat
        if isinstance(other, DMat):
            return DMat(
                other.arr + self.get_dense()
            )
        else:
            return NotImplemented

    def __sub__(self, other):
        # LRMat - DMat
        # Note that this should only happen on the deepest level
        if isinstance(other, DMat):
            return DMat(
                self.get_dense() - other.arr
            )
        # LRMat - LRMat
        elif isinstance(other, LRMat):
            return LRMat(
                self.get_dense() - other.get_dense(),
                k=self.k,
                p=self.p
            )
        else:
            return NotImplemented

    def __neg__(self):
        return LRMat(
            [
                self[0],
                -self[1],
                self[2]
            ],
            k=self.k,
            p=self.p
        )
