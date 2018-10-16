.SUFFIXES: .cpp .cu

CXX = mpicxx -std=c++11 -ggdb3 -O3 -fopenmp -I. -Wall -Wfatal-errors
NVCC = nvcc -ccbin=g++-5 -std=c++11 -I. -arch sm_35 -Xcompiler "-fopenmp -Wall -Wfatal-errors"

SOURCES = errors.o print.o mpi_utils.o timer.o functions.o any.o node.o dense.o low_rank.o hierarchical.o
TEST_SOURCES = test/test_helper.o test/test_mpi_block_creation.o test/test_mpi_dense_lu.o

.cpp.o:
	$(CXX) -c $? -o $@

.cu.o:
	$(NVCC) -c $? -o $@

all:
	make rsvd block_lu blr_lu h_lu gpu blr_lu_gpu h_lu_gpu

lib: $(SOURCES)
	ar -cr libhicma.a $(SOURCES)
	ranlib libhicma.a

rsvd: rsvd.o batch_rsvd.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	./a.out

block_lu: block_lu.o batch_rsvd.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	./a.out

blr_lu: blr_lu.o batch_rsvd.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	./a.out

h_lu: h_lu.o batch_rsvd.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	./a.out 6

gpu: gpu.o batch_rsvd_gpu.o $(SOURCES)
	$(CXX) $? -L/home/rioyokota/magma-2.3.0/lib -lm -lkblas-gpu -lmagma -lcusparse -lcublas -lcudart -lblas -llapacke -lpthread -lm -ldl -lstdc++ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
	./a.out

blr_lu_gpu: blr_lu.o batch_rsvd_gpu.o $(SOURCES)
	$(CXX) $? -L/home/rioyokota/magma-2.3.0/lib -lm -lkblas-gpu -lmagma -lcusparse -lcublas -lcudart -lblas -llapacke -lpthread -lm -ldl -lstdc++ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
	./a.out

h_lu_gpu: h_lu.o batch_rsvd_gpu.o $(SOURCES)
	$(CXX) $? -L/home/rioyokota/magma-2.3.0/lib -lm -lkblas-gpu -lmagma -lcusparse -lcublas -lcudart -lblas -llapacke -lpthread -lm -ldl -lstdc++ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
	./a.out 6

test:  $(TEST_SOURCES) $(SOURCES)
	$(CXX) $? -lblas -llapacke
	mpirun -np 4 ./a.out

clean:
	$(RM) *.o *.out *.xml
