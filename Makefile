CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -I.

.cxx.o:
	$(CXX) -c $? -o $@

block_lu: block_lu.o
	$(CXX) $? -lblas -llapack -lgsl -lgslcblas -lm
	valgrind ./a.out

blr_lu: blr_lu.o
	$(CXX) $? -lblas -llapack -lgsl -lgslcblas -lm
	valgrind ./a.out

id: id.o
	$(CXX) $? -lgsl -lgslcblas -lm
	./a.out

clean:
	$(RM) *.o *.out
