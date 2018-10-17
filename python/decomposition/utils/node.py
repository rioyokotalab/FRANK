#!/bin/python


class Node(object):
    """
    Basic matrix class with interactions producing no change.
    """
    def __init__(self, parent=None, pos=None):
        """
        Init the block matrix from the np.array arr

        Arguments
        ---------
        self - np.array
        parent - HMat
            Parent of the HMat to be created.
        pos - tuple
            Holding the row and column index that this object should have
            within its parent.
        """
        self.parent = parent
        if self.parent is None:
            self.root = True
            self.level = 0
        else:
            self.root = False
            self.level = self.parent.level + 1
        self.row = None
        self.col = None
        self._init_position(pos)

    def _init_position(self, pos):
        """
        Initialize the position data of the HMat.

        Arguments
        ---------
        pos - tuple
            Holding the row and column index that this object should have
            within its parent.
        """
        if self.root:
            assert pos is None
            self.pos = (0, 0)
            self.row = 0
            self.col = 0
        else:
            assert pos is not None
            self.pos = pos
            # Calculate total position using parent position
            # Equivalent to Morton indexing
            self.row = (self.parent.row << 1) + pos[0]
            self.col = (self.parent.col << 1) + pos[1]

    def update_position(self, parent, pos):
        """
        Recursively update the position data of the HMat.
        """
        self.parent = parent
        self.level = self.parent.level
        self.root = self.parent is None
        self._init_position(pos)
