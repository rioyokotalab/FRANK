CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -I.

.cxx.o:
	$(CXX) -c $? -o $@

block_lu: block_lu.o node.o dense.o low_rank.o hierarchical.o id.o
	$(CXX) $? -lblas -llapack -lgsl -lgslcblas -lm
	valgrind ./a.out

blr_lu: blr_lu.o node.o dense.o low_rank.o hierarchical.o id.o
	$(CXX) $? -lblas -llapack -lgsl -lgslcblas -lm
	valgrind ./a.out

hodlr_lu: hodlr_lu.o node.o dense.o low_rank.o hierarchical.o id.o
	$(CXX) $? -lblas -llapack -lgsl -lgslcblas -lm
	valgrind ./a.out

id_test: id_test.o id.o
	$(CXX) $? -lgsl -lgslcblas -lm
	valgrind ./a.out

clean:
	$(RM) *.o *.out
