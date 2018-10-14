#include "functions.h"
#include "low_rank.h"
#include "timer.h"

#include <cublas_v2.h>
#include <magma_v2.h>
#include "kblas.h"
#include "batch_rand.h"
#include "batch_ara.h"

// Are all of these needed?
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/logical.h>

extern "C" void gpuAssert(cudaError_t code, const char *file, int line) {
  if(code != cudaSuccess) {
    printf("gpuAssert: %s(%d) %s %d\n", cudaGetErrorString(code), (int)code, file, line);
    exit(-1);
  }
}

#define check_error(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define TESTING_MALLOC_DEV( ptr, T, size) check_error( cudaMalloc( (void**)&ptr, (size)*sizeof(T) ) )

#define COPY_DATA_UP(d_A, h_A, el_size, el_type)                        \
  cudaSetDevice(0);                                                     \
  check_error( cudaMemcpy(d_A, h_A, el_size * sizeof(el_type), cudaMemcpyHostToDevice) ); \
  cudaDeviceSynchronize();

#define COPY_DATA_DOWN(h_A, d_A, el_size, el_type)                      \
  cudaSetDevice(0);                                                     \
  check_error( cudaMemcpy(h_A, d_A, el_size * sizeof(el_type), cudaMemcpyDeviceToHost ) ); \
  cudaDeviceSynchronize();

#define TESTING_FREE_DEV(ptr)   check_error( cudaFree( (ptr) ) );

template<class T>
struct UnaryAoAAssign : public thrust::unary_function<int, T*> {
  T* original_array;
  int stride;
  UnaryAoAAssign(T* original_array, int stride) { this->original_array = original_array; this->stride = stride; }
  __host__ __device__
  T* operator()(const unsigned int& thread_id) const { return original_array + thread_id * stride; }
};

template<class T>
void generateArrayOfPointersT(T* original_array, T** array_of_arrays, int stride,
                              int num_arrays, cudaStream_t stream
                              ) {
  thrust::device_ptr<T*> dev_data(array_of_arrays);
  thrust::transform(
                    thrust::cuda::par.on(stream),
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_arrays),
                    dev_data,
                    UnaryAoAAssign<T>(original_array, stride)
                    );
  check_error( cudaGetLastError() );
}
extern "C" void generateDArrayOfPointers(double* original_array, double** array_of_arrays,
                                         int stride, int num_arrays, cudaStream_t stream) {
  generateArrayOfPointersT<double>(original_array, array_of_arrays, stride, num_arrays, stream);
}

using namespace hicma;

int main(int argc, char** argv) {
  int N = 8; // 32
  int rank = 6; // 16
  std::vector<double> randx(2*N);
  for (int i=0; i<2*N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  print("Time");
  start("Init matrix");
  Dense A(laplace1d, randx, N, N-2, 0, N);
  stop("Init matrix");
  start("Randomized SVD");
  LowRank LR(A.dim[0],A.dim[1],rank);

  int batchCount = 1;
  kblasHandle_t handle;
  kblasRandState_t rand_state;
  kblasCreate(&handle);
  kblasInitRandState(handle, &rand_state, 16384*2, 0);
  kblasEnableMagma(handle);
  magma_init();
  int *d_m, *d_n, *d_lda, *d_ldu, *d_ldv, *d_rank;
  double *d_A, *d_U, *d_V;
  double **d_A_ptrs, **d_U_ptrs, **d_V_ptrs;
  TESTING_MALLOC_DEV(d_m, int, batchCount);
  TESTING_MALLOC_DEV(d_n, int, batchCount);
  TESTING_MALLOC_DEV(d_lda, int, batchCount);
  TESTING_MALLOC_DEV(d_ldu, int, batchCount);
  TESTING_MALLOC_DEV(d_ldv, int, batchCount);
  TESTING_MALLOC_DEV(d_rank, int, batchCount);
  TESTING_MALLOC_DEV(d_A, double, A.dim[0] * A.dim[1] * batchCount);
  TESTING_MALLOC_DEV(d_U, double, A.dim[0] * rank * batchCount);
  TESTING_MALLOC_DEV(d_V, double, A.dim[1] * rank * batchCount);
  TESTING_MALLOC_DEV(d_A_ptrs, double*, batchCount);
  TESTING_MALLOC_DEV(d_U_ptrs, double*, batchCount);
  TESTING_MALLOC_DEV(d_V_ptrs, double*, batchCount);
  generateDArrayOfPointers(d_A, d_A_ptrs, A.dim[0] * A.dim[1], batchCount, 0);
  generateDArrayOfPointers(d_U, d_U_ptrs, A.dim[0] * rank, batchCount, 0);
  generateDArrayOfPointers(d_V, d_V_ptrs, A.dim[1] * rank, batchCount, 0);
  Dense At = A;
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<A.dim[1]; j++) {
      At.data[j*A.dim[0]+i] = A(i,j);
    }
  }
  COPY_DATA_UP(d_m, &A.dim[0], batchCount, int);
  COPY_DATA_UP(d_n, &A.dim[1], batchCount, int);
  COPY_DATA_UP(d_lda, &A.dim[0], batchCount, int);
  COPY_DATA_UP(d_ldu, &A.dim[0], batchCount, int);
  COPY_DATA_UP(d_ldv, &A.dim[1], batchCount, int);
  COPY_DATA_UP(d_A, &At[0], A.dim[0] * A.dim[1], double);
  double tol = 1e-6;
  int block_size = 32;
  int ara_r = 10;
  kblas_ara_batch_wsquery<double>(handle, block_size, batchCount);
  kblasAllocateWorkspace(handle);
  kblas_ara_batched(
                    handle, d_m, d_n, d_A_ptrs, d_lda, d_U_ptrs, d_ldu, d_V_ptrs, d_ldv, d_rank,
                    tol, A.dim[0], A.dim[1], rank, block_size, ara_r, rand_state, batchCount
                    );
  Dense U(A.dim[0],rank);
  Dense Ut = U;
  COPY_DATA_DOWN(&Ut[0], d_U, U.dim[0] * U.dim[1], double);
  for (int i=0; i<U.dim[0]; i++) {
    for (int j=0; j<U.dim[1]; j++) {
      U(i,j) = Ut.data[j*U.dim[0]+i];
    }
  }
  Dense V(A.dim[1],rank);
  Dense Vt = V;
  COPY_DATA_DOWN(&Vt[0], d_V, V.dim[0] * V.dim[1], double);
  for (int i=0; i<V.dim[0]; i++) {
    for (int j=0; j<V.dim[1]; j++) {
      V(i,j) = Vt.data[j*V.dim[0]+i];
    }
  }
  COPY_DATA_DOWN(&rank, d_rank, batchCount, int);
  U.resize(A.dim[0],rank);
  V.resize(A.dim[1],rank);
  Dense A2(A);
  A2.gemm(U, V, CblasNoTrans, CblasTrans, 1, 0);
  stop("Randomized SVD");
  double diff = (A - A2).norm();
  double norm = A.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  TESTING_FREE_DEV(d_m);
  TESTING_FREE_DEV(d_n);
  TESTING_FREE_DEV(d_lda);
  TESTING_FREE_DEV(d_ldu);
  TESTING_FREE_DEV(d_ldv);
  TESTING_FREE_DEV(d_rank);
  TESTING_FREE_DEV(d_A);
  TESTING_FREE_DEV(d_U);
  TESTING_FREE_DEV(d_V);
  kblasDestroy(&handle);
  return 0;
}
