CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -I.

.cxx.o:
	$(CXX) -c $? -o $@

test: test.o
	$(CXX) $?
	valgrind ./a.out

block_lu: block_lu.o
	$(CXX) $? -lblas -llapack
	valgrind ./a.out

blr_lu: blr_lu.o
	$(CXX) $? -lblas -llapack -lgsl -lgslcblas -lm
	valgrind ./a.out

low_rank: id.o
	$(CXX) $? -lgsl -lgslcblas -lm
	./a.out

clean:
	$(RM) *.o *.out
