#!/bin/python
import numpy as np
from scipy.linalg import solve_triangular

from decomposition.utils.node import Node
from decomposition.lu import lup_factorize


class DMat(Node):
    """
    Dense matrix container.
    """
    # TODO Pass generator function
    def __init__(self, arr, parent=None, pos=None):
        super().__init__(parent, pos)
        assert isinstance(arr, np.ndarray)
        self.arr = arr
        self.m, self.n = arr.shape

    def get_dense(self):
        return self.arr

    def _get_lower_triangle(self):
        return DMat(
            np.tril(self.arr, -1) + np.identity(self.arr.shape[0])
        )

    def _get_upper_triangle(self):
        return DMat(
            np.triu(self.arr)
        )

    def getrf(self):
        lup_factorize(self.arr)

    def upper_trsm(self, other):
        a = self.get_dense()
        b = other.get_dense().transpose()
        x = solve_triangular(
            a,
            b,
            lower=False,
            unit_diagonal=False,
            overwrite_b=True,
            trans='T'
        ).transpose()
        return DMat(x)

    def lower_trsm(self, other):
        a = self.get_dense()
        b = other.get_dense()
        x = solve_triangular(
            a,
            b,
            lower=True,
            unit_diagonal=True,
            overwrite_b=True,
            trans=0
        )
        return DMat(x)

    def _upper(self):
        return DMat(self.arr[:int(self.m/2), :])

    def _lower(self):
        return DMat(self.arr[int(self.m/2):, :])

    def _left(self):
        return DMat(self.arr[:, :int(self.n/2)])

    def _right(self):
        return DMat(self.arr[:, int(self.n/2):])

    def __matmul__(self, other):
        # DMat @ DMat
        if isinstance(other, DMat):
            return DMat(self.arr @ other.arr)
        else:
            return NotImplemented

    def __sub__(self, other):
        # DMat - DMat
        if isinstance(other, DMat):
            return DMat(self.arr - other.arr)
        else:
            return NotImplemented

    def __add__(self, other):
        # DMat + DMat
        if isinstance(other, DMat):
            return DMat(self.arr + other.arr)
        else:
            return NotImplemented

    def __neg__(self):
        # - DMat
        return DMat(-self.arr)
