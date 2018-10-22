#!/bin/python


class Node(object):
    """
    Basic matrix class with interactions producing no change.
    """
    def __init__(self, i_abs, j_abs, level):
        """
        Init the block matrix from the np.array arr

        Arguments
        ---------
        self - np.array
        """
        self.i_abs = i_abs
        self.j_abs = j_abs
        self.level = level

    def type(self):
        return 'Node'
