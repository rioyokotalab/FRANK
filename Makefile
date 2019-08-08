.SUFFIXES: .cpp .cu

CXX = g++ -std=c++11 -ggdb3 -O3 -fopenmp -I. -Wall -Wfatal-errors #-DUSE_MKL
NVCC = nvcc -ccbin=g++-5 -std=c++11 -I. -arch sm_35 -Xcompiler "-ggdb3 -fopenmp -Wall -Wfatal-errors"

SOURCES = print.o timer.o functions.o any.o node.o dense.o low_rank.o hierarchical.o

.cpp.o:
	$(CXX) -c $? -o $@

.cu.o:
	$(NVCC) -c $? -o $@

all:
	make cpu gpu

cpu:
	make rsvd block_lu blr_lu h_lu

gpu:
	make rsvd_gpu gemm_gpu block_lu_gpu blr_lu_gpu h_lu_gpu

lib: batch.o $(SOURCES)
	ar -cr libhicma.a batch.o $(SOURCES)
	ranlib libhicma.a

rsvd: rsvd.o batch.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

block_lu: block_lu.o batch.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

blr_lu: blr_lu.o batch.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

h_lu: h_lu.o batch.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out 6

block_qr: block_qr.o batch.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

tiled_qr: tiled_qr.o batch.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	./a.out

blr_qr: blr_qr.o batch.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out

tlr_qr: tlr_qr.o batch.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	./a.out

h_qr: h_qr.o batch.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	valgrind ./a.out 6

htl_qr: htl_qr.o batch.o $(SOURCES)
	$(CXX) $? -lblas -llapacke
	./a.out 1

rsvd_gpu: rsvd_gpu.o batch_gpu.o $(SOURCES)
	$(CXX) $? -L/home/rioyokota/magma-2.3.0/lib -lm -lkblas-gpu -lmagma -lcusparse -lcublas -lcudart -lblas -llapacke -lpthread -lm -ldl -lstdc++ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
	./a.out

gemm_gpu: gemm_gpu.o batch_gpu.o $(SOURCES)
	$(CXX) $? -L/home/rioyokota/magma-2.3.0/lib -lm -lkblas-gpu -lmagma -lcusparse -lcublas -lcudart -lblas -llapacke -lpthread -lm -ldl -lstdc++ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
	./a.out

block_lu_gpu: block_lu.o batch_gpu.o $(SOURCES)
	$(CXX) $? -L/home/rioyokota/magma-2.3.0/lib -lm -lkblas-gpu -lmagma -lcusparse -lcublas -lcudart -lblas -llapacke -lpthread -lm -ldl -lstdc++ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
	./a.out

blr_lu_gpu: blr_lu.o batch_gpu.o $(SOURCES)
	$(CXX) $? -L/home/rioyokota/magma-2.3.0/lib -lm -lkblas-gpu -lmagma -lcusparse -lcublas -lcudart -lblas -llapacke -lpthread -lm -ldl -lstdc++ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
	./a.out

h_lu_gpu: h_lu.o batch_gpu.o $(SOURCES)
	$(CXX) $? -L/home/rioyokota/magma-2.3.0/lib -lm -lkblas-gpu -lmagma -lcusparse -lcublas -lcudart -lblas -llapacke -lpthread -lm -ldl -lstdc++ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
	./a.out 6

clean:
	$(RM) *.o *.out *.xml
