#include "functions.h"
#include "low_rank.h"
#include "timer.h"

#include <cublas_v2.h>
#include <magma_v2.h>
#include "kblas.h"

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
  int k = 4; // 16
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
  LowRank LR(A.dim[0],A.dim[1],k);
  int rank = k; // k + 5
  LR.S = Dense(rank, rank);
  Dense RN(A.dim[1],rank);
  std::mt19937 generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  for (int i=0; i<A.dim[1]*rank; i++) {
    RN[i] = distribution(generator);
  }
  Dense Y(A.dim[0],rank);
  Dense Q(A.dim[0],rank);
  Dense R(rank,rank);
  Dense Bt(A.dim[1],rank);
  Dense Qb(A.dim[1],rank);
  Dense Rb(rank,rank);
  Dense Ur(rank,rank);
  Dense Vr(rank,rank);

  int batchCount = 1;
  kblasHandle_t handle;
  kblasCreate(&handle);
  kblasEnableMagma(handle); // Is this needed?
  magma_init();
  double *d_Y, *d_A, *d_RN;
  double **d_Y_ptrs, **d_A_ptrs, **d_RN_ptrs;
  TESTING_MALLOC_DEV(d_Y, double, Y.dim[0] * Y.dim[1]);
  TESTING_MALLOC_DEV(d_A, double, A.dim[0] * A.dim[1]);
  TESTING_MALLOC_DEV(d_RN, double, RN.dim[0] * RN.dim[1]);
  TESTING_MALLOC_DEV(d_Y_ptrs, double*, batchCount);
  TESTING_MALLOC_DEV(d_A_ptrs, double*, batchCount);
  TESTING_MALLOC_DEV(d_RN_ptrs, double*, batchCount);
  generateDArrayOfPointers(d_Y, d_Y_ptrs, Y.dim[0] * Y.dim[1], batchCount, 0);
  generateDArrayOfPointers(d_A, d_A_ptrs, A.dim[0] * A.dim[1], batchCount, 0);
  generateDArrayOfPointers(d_RN, d_RN_ptrs, RN.dim[0] * RN.dim[1], batchCount, 0);
  Dense At = A;
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<A.dim[1]; j++) {
      At.data[j*A.dim[0]+i] = A(i,j);
    }
  }
  Dense RNt = RN;
  for (int i=0; i<RN.dim[0]; i++) {
    for (int j=0; j<RN.dim[1]; j++) {
      RNt.data[j*RN.dim[0]+i] = RN(i,j);
    }
  }
  COPY_DATA_UP(d_A, &At[0], A.dim[0] * A.dim[1], double);
  COPY_DATA_UP(d_RN, &RNt[0], RN.dim[0] * RN.dim[1], double);
  kblas_gemm_batch(
                   handle, KBLAS_NoTrans, KBLAS_NoTrans, Y.dim[0], Y.dim[1], A.dim[1], 1.0,
                   (const double**)d_A_ptrs, A.dim[0], (const double**)d_RN_ptrs, RN.dim[0], 0.0,
                   d_Y_ptrs, Y.dim[0], batchCount
                   );
  Dense Yt = Y;
  COPY_DATA_DOWN(&Yt[0], d_Y, Y.dim[0] * Y.dim[1], double);
  for (int i=0; i<Y.dim[0]; i++) {
    for (int j=0; j<Y.dim[1]; j++) {
      Y(i,j) = Yt.data[j*Y.dim[0]+i];
    }
  }
  TESTING_FREE_DEV(d_Y);
  TESTING_FREE_DEV(d_A);
  TESTING_FREE_DEV(d_RN);
  TESTING_FREE_DEV(d_Y_ptrs);
  TESTING_FREE_DEV(d_A_ptrs);
  TESTING_FREE_DEV(d_RN_ptrs);
  kblasDestroy(&handle);
  Y.print();
  Y.gemm(A, RN, CblasNoTrans, CblasNoTrans, 1, 0);
  Y.print();
  Y.qr(Q, R);
  Bt.gemm(A, Q, CblasTrans, CblasNoTrans, 1, 0);
  Bt.qr(Qb,Rb);
  Rb.svd(Vr,LR.S,Ur);
  Ur.resize(k,rank);
  LR.U.gemm(Q, Ur, CblasNoTrans, CblasTrans, 1, 0);
  Vr.resize(rank,k);
  LR.V.gemm(Vr, Qb, CblasTrans, CblasTrans, 1, 0);
  LR.S.resize(k,k);
  stop("Randomized SVD");
  double diff = (A - Dense(LR)).norm();
  double norm = A.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
