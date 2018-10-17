#!/bin/python
import numpy as np

from decomposition.qr import (
    house_qr,
    get_q_r_p
)

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


def main():
    m = 6
    n = 6
    array = np.random.random_integers(0, 10, (m, n)).astype(float)

    print('Input array:\n', array)
    house_qr(array)
    print('Result:\n', array)
    Q, R, _ = get_q_r_p(array)
    print('Control:\n', Q @ R)


if __name__ == '__main__':
    main()
