#!/bin/python


def make_block(arr, block_s):
    """
    Split arr into blocks of size block_s.

    ## NOT IMPLEMENTED YET!!
    arr is a (m, n) array. If m is not divisible by block_s, the right
    outermost blocks will have n < block_s. If n is not divisible, the bottom
    outermost blocks will have m > block_s.
    """
    m, n = arr.shape
    rows = int(m/block_s)
    cols = int(n/block_s)
    assert m == n
    assert m % block_s == 0
    arr_b = [[]*cols]*rows
    for i in range(0, rows):
        row = []
        for j in range(0, cols):
            entry = arr[i*block_s:(i+1)*block_s, j*block_s:(j+1)*block_s]
            row.append(entry)
        arr_b[i] = row
    return arr_b
