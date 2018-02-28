.SUFFIXES: .cxx .o

CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -I.

.cxx.o	:
	$(CXX) -c $? -o $@

block_lu: block_lu.o
	$(CXX) $? -lblas -llapack
	valgrind ./a.out

clean:
	$(RM) *.o *.out
