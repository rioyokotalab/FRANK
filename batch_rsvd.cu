#include "low_rank.h"
#include "batch_rsvd.h"

#include <cublas_v2.h>
#include <magma_v2.h>
#include "kblas.h"
#include "batch_rand.h"
#include "batch_ara.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace hicma {

  bool useBatch;
  std::vector<int> h_m;
  std::vector<int> h_n;
  std::vector<Dense> vecA;
  std::vector<Any*> vecLR;

  struct Assign : public thrust::unary_function<int, double*> {
    double* original_array;
    int stride;
    Assign(double* original_array, int stride) {
      this->original_array = original_array;
      this->stride = stride;
    }
    __host__ __device__ double* operator()(const unsigned int& thread_id) const {
      return original_array + thread_id * stride;
    }
  };

  void ArrayOfPointers(
                       double* original_array, double** array_of_arrays, int stride,
                       int num_arrays, cudaStream_t stream
                       ) {
    thrust::device_ptr<double*> dev_data(array_of_arrays);
    thrust::transform(
                      thrust::cuda::par.on(stream),
                      thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(num_arrays),
                      dev_data,
                      Assign(original_array, stride)
                      );
  }

  void batch_rsvd() {
    int batchCount = h_m.size();
    if (batchCount == 0) return;
    kblasHandle_t handle;
    kblasRandState_t rand_state;
    kblasCreate(&handle);
    kblasInitRandState(handle, &rand_state, 16384*2, 0);
    kblasEnableMagma(handle);
    magma_init();
    int *d_m, *d_n, *d_k;
    double *d_A, *d_U, *d_V;
    double **p_A, **p_U, **p_V;
    int max_m = 0, max_n = 0;
    for (int b=0; b<batchCount; b++) {
      max_m = std::max(max_m, h_m[b]);
      max_n = std::max(max_n, h_n[b]);
    }
    int max_k = max_n;
    std::vector<double> h_A(max_m * max_n * batchCount);
    std::vector<double> h_U(max_m * max_n * batchCount);
    std::vector<double> h_V(max_n * max_n * batchCount);
    cudaMalloc( (void**)&d_m, batchCount * sizeof(int) );
    cudaMalloc( (void**)&d_n, batchCount * sizeof(int) );
    cudaMalloc( (void**)&d_k, batchCount * sizeof(int) );
    cudaMalloc( (void**)&d_A, h_A.size() * sizeof(double) );
    cudaMalloc( (void**)&d_U, h_U.size() * sizeof(double) );
    cudaMalloc( (void**)&d_V, h_V.size() * sizeof(double) );
    cudaMalloc( (void**)&p_A, batchCount * sizeof(double*) );
    cudaMalloc( (void**)&p_U, batchCount * sizeof(double*) );
    cudaMalloc( (void**)&p_V, batchCount * sizeof(double*) );
    ArrayOfPointers(d_A, p_A, max_m * max_n, batchCount, 0);
    ArrayOfPointers(d_U, p_U, max_m * max_n, batchCount, 0);
    ArrayOfPointers(d_V, p_V, max_n * max_n, batchCount, 0);
    for (int b=0; b<batchCount; b++) {
      for (int i=0; i<vecA[b].dim[0]; i++) {
        for (int j=0; j<vecA[b].dim[1]; j++) {
          h_A[i+j*max_m+b*max_m*max_n] = vecA[b](i,j);
        }
      }
    }
    cudaMemcpy(d_m, &h_m[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &h_n[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, &h_A[0], h_A.size() * sizeof(double), cudaMemcpyHostToDevice);
    double tol = 1e-6;
    int block_size = 32;
    int ara_r = 10;
    kblas_ara_batch_wsquery<double>(handle, block_size, batchCount);
    kblasAllocateWorkspace(handle);
    kblas_ara_batched(
                      handle, d_m, d_n, p_A, d_m, p_U, d_m, p_V, d_n, d_k,
                      tol, max_m, max_n, max_k, block_size, ara_r, rand_state, batchCount
                      );
    std::vector<int> h_k(batchCount);
    cudaMemcpy(&h_k[0], d_k, batchCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_U[0], d_U, h_U.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_V[0], d_V, h_V.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (int b=0; b<batchCount; b++) {
      Dense A = vecA[b];
      LowRank LR(A.dim[0], A.dim[1], h_k[b], A.i_abs, A.j_abs, A.level);
      for (int i=0; i<LR.dim[0]; i++) {
        for (int j=0; j<LR.rank; j++) {
          LR.U(i,j) = h_U[i+j*max_m+b*max_m*max_n];
        }
      }
      for (int i=0; i<LR.rank; i++) {
        for (int j=0; j<LR.dim[1]; j++) {
          LR.V(i,j) = h_V[j+i*max_n+b*max_n*max_n];
        }
        LR.S(i,i) = 1;
      }
      *vecLR[b] = LR;
    }
    cudaFree(p_A);
    cudaFree(p_U);
    cudaFree(p_V);
    cudaFree(d_A);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_k);
    kblasDestroy(&handle);
  }
}
