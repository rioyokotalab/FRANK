CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -I.

SOURCES = hblas.o dense.o id.o low_rank.o hierarchical.o

.cpp.o:
	$(CXX) -c $? -o $@

block_lu: block_lu.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

blr_lu: blr_lu.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

h_lu: h_lu.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out 1

clean:
	$(RM) *.o *.out
