from timeit import default_timer as timer

from hicma.print import print_time


tic = {}

def start(event):
    tic[event] = timer()

def stop(event):
    toc = timer()
    print_time(event, toc - tic[event])
