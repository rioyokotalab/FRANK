.SUFFIXES: .cxx .o

CC = gcc -Wall -O3 -I.
CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -I.

.cxx.o:
	$(CXX) -c $? -o $@

block_lu: block_lu.o
	$(CXX) $? -lblas -llapack
	valgrind ./a.out

block_lu_mpi: block_lu_mpi.o
	$(CXX) $? -lblas -llapack -lblacsCinit-openmpi -lblacs-openmpi
	mpirun -np 4 ./a.out

low_rank: driver_single_core_gsl.o matrix_vector_functions_gsl.o low_rank_svd_algorithms_gsl.o
	$(CC) $? -lgsl -lgslcblas -lm
	./a.out 0
	./a.out 1
	./a.out 2
	./a.out 3
	./a.out 4
	./a.out 5

clean:
	$(RM) *.o *.out
