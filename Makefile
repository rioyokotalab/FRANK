.SUFFIXES: .cxx .o

CC = gcc -Wall -Wunused-function -O3 -I.
CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -I.

.cxx.o:
	$(CXX) -c $? -o $@

block_lu: block_lu.o
	$(CXX) $? -lblas -llapack
	valgrind ./a.out

block_lu_mpi: block_lu_mpi.o
	$(CXX) $? -lblas -llapack -lblacsCinit-openmpi -lblacs-openmpi
	mpirun -np 4 ./a.out

low_rank: id.o
	$(CC) $? -lgsl -lgslcblas -lm
	./a.out 0
	./a.out 1
	./a.out 2
	./a.out 3
	./a.out 4
	./a.out 5

clean:
	$(RM) *.o *.out
