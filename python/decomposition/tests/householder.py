#!/bin/python
import numpy as np

from householder import get_vector

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


x = np.array([4, 5, 6, 7]).astype(float)

v, beta = get_vector(x)

H = np.identity(x.shape[0]) - beta * np.outer(v, v)
print('Householder matrix:\n', H)
print('New x:\n', H @ x)
