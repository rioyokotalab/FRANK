def printf(s, n=None):
    if n is None:
        s = '--- ' + s + ' '
        print("{:-<37}".format(s))
    else:
        print('{:25} : {:3.1e}'.format(s, n))


def print_time(s, n):
    print('{:25} : {:0<9f}'.format(s, n))
