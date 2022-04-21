import numpy as np
import random

class Rand48(object):
    def __init__(self, seed):
        self.n = seed
    def seed(self, seed):
        self.n = seed
    def srand(self, seed):
        self.n = (seed << 16) + 0x330e
    def next(self):
        self.n = (25214903917 * self.n + 11) & (2**48 - 1)
        return self.n
    def drand(self):
        return self.next() / 2**48
    def lrand(self):
        return self.next() >> 17
    def mrand(self):
        n = self.next() >> 16
        if n & (1 << 31):
            n -= 1 << 32
        return n

def zeros(data, x, ni, nj, i_begin, j_begin):
    data.reshape((ni, nj))
    for i in range(ni):
        for j in range(nj):
            data[i, j] = 0


def rand_data(data, x, ni, nj, i_begin, j_begin):
    data.reshape((ni, nj))
    for i in range(ni):
        for j in range(nj):
            data[i, j] = 0.5 # RANDOM HERE


def arange(data, x, ni, nj, i_begin, j_begin):
    data.reshape((ni, nj))
    for i in range(ni):
        for j in range(nj):
            data[i, j] = i*nj + j


def laplace1d(data, x, ni, nj, i_begin, j_begin):
    data.reshape((ni, nj))
    for i in range(ni):
        for j in range(nj):
            data[i, j] = 1. / (abs(x[i + i_begin] - x[j + j_begin]) + 1e-3)

def identity(data, x, ni, nj, i_begin, j_begin):
    # Expects matrix as x, just copies in
    data.reshape((ni, nj))
    for i in range(ni):
        for j in range(nj):
            data[i, j] = x[i_begin+i, j_begin+j]
