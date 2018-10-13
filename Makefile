.SUFFIXES: .cpp .cu

CXX = mpicxx -std=c++11 -ggdb3 -O3 -fopenmp -I. -Wall -Wfatal-errors
NVCC = nvcc -ccbin=g++-5 -std=c++11 -I. -arch sm_35 -Xcompiler "-fopenmp -Wall -Wfatal-errors"

SOURCES = errors.o print.o mpi_utils.o timer.o functions.o dense.o low_rank.o hierarchical.o node.o any.o
TEST_SOURCES = test/test_helper.o test/test_mpi_block_creation.o test/test_mpi_dense_lu.o

.cpp.o:
	$(CXX) -c $? -o $@

.cu.o:
	$(NVCC) -c $? -o $@

rsvd: rsvd.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

gpu: gpu.o $(SOURCES)
	$(CXX) $? -L/home/rioyokota/magma-2.3.0/lib -lm -lkblas-gpu -lmagma -lcusparse -lcublas -lcudart -lblas -llapacke -lpthread -lm -ldl -lstdc++
	./a.out

block_lu: block_lu.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

blr_lu: blr_lu.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

h_lu: h_lu.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out 6

test:  $(TEST_SOURCES) $(SOURCES)
	$(CXX) $? -lblas -llapacke
	mpirun -np 4 ./a.out

clean:
	$(RM) *.o *.out *.xml
