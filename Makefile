CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -I.

.cxx.o:
	$(CXX) -c $? -o $@

block_lu: block_lu.o hblas.o dense.o id.o low_rank.o hierarchical.o
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

blr_lu: blr_lu.o hblas.o dense.o id.o low_rank.o hierarchical.o
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

hodlr_lu: hodlr_lu.o hblas.o dense.o id.o low_rank.o hierarchical.o
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

clean:
	$(RM) *.o *.out
