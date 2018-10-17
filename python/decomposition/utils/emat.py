#!/bin/python
import numpy as np

from decomposition.utils.node import Node


class EMat(Node):
    """
    Empty matrix object interacting like a null matrix.
    """
    def __matmul__(self, other):
        """
        Multiplications with this class return a null matrix.
        """
        if isinstance(other, Node):
            return self
        elif isinstance(other, np.ndarray):
            return self
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """
        Right multiplication with this class return a null matrix.
        """
        if isinstance(other, Node):
            return self
        elif isinstance(other, np.ndarray):
            return self
        else:
            return NotImplemented

    def __add__(self, other):
        """
        Addition with this class returns the right summand.
        """
        if isinstance(other, Node):
            return other
        elif isinstance(other, np.ndarray):
            return other
        else:
            return NotImplemented

    def __radd__(self, other):
        """
        Right addition with this class return the left summand
        """
        if isinstance(other, Node):
            return other
        elif isinstance(other, np.ndarray):
            return other
        else:
            return NotImplemented

    def __sub__(self, other):
        """
        Subtraction from this class the negative of the other object.
        """
        if isinstance(other, Node):
            return -other
        elif isinstance(other, np.ndarray):
            return -other
        else:
            return NotImplemented

    def __rsub__(self, other):
        """
        Right subtraction with this class returns the other object..
        """
        if isinstance(other, Node):
            return other
        elif isinstance(other, np.ndarray):
            return other
        else:
            return NotImplemented

    def __neg__(self):
        """
        Negation of this class produces the class itself.
        """
        return self
