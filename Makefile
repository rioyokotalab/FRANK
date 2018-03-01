.SUFFIXES: .cxx .o

CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -I.

.cxx.o	:
	$(CXX) -c $? -o $@

block_lu: block_lu.o
	$(CXX) $? -lblas -llapack
	valgrind ./a.out

block_lu_mpi: block_lu_mpi.o
	$(CXX) $? -lblas -llapack -lblacsCinit-openmpi -lblacs-openmpi
	mpirun -np 4 ./a.out

clean:
	$(RM) *.o *.out
