CXX = mpicxx -std=c++14 -ggdb3 -Wall -O3 -fopenmp -I.

SOURCES = dense.o id.o low_rank.o hierarchical.o node.o block.o

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
	valgrind ./a.out 4

clean:
	$(RM) *.o *.out
