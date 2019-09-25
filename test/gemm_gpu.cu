#include "any.h"
#include "low_rank.h"
#include "hierarchical.h"
#include "functions.h"
#include "print.h"
#include "timer.h"

#include <cublas_v2.h>
#include <magma_v2.h>
#include "kblas.h"
#include "testing_helper.h"

using namespace hicma;

int main(int argc, char** argv) {
  int batchCount = 4;
  double alpha = 0.28;
  double beta = 1.2;
  std::vector<int> h_m(batchCount);
  std::vector<int> h_n(batchCount);
  std::vector<int> h_k(batchCount);
  int N = 512;
  int max_m = 0, max_n = 0, max_k = 0;
  for(int b=0; b<batchCount; b++){
    h_m[b] = 1 + (rand() % N);
    h_n[b] = 1 + (rand() % N);
    h_k[b] = 1 + (rand() % N);
    max_m = std::max(max_m,h_m[b]);
    max_n = std::max(max_n,h_n[b]);
    max_k = std::max(max_k,h_k[b]);
  }
  std::vector<Dense> vecA;
  std::vector<Dense> vecB;
  std::vector<Dense> vecC;
  for (int b=0; b<batchCount; b++) {
    std::vector<double> x;
    Dense A(random, x, h_m[b], h_k[b], 0, 0);
    Dense B(random, x, h_k[b], h_n[b], 0, 0);
    Dense C(random, x, h_m[b], h_n[b], 0, 0);
    vecA.push_back(A);
    vecB.push_back(B);
    vecC.push_back(C);
  }
  std::vector<double> h_A(max_m * max_k * batchCount);
  std::vector<double> h_B(max_k * max_n * batchCount);
  std::vector<double> h_C(max_m * max_n * batchCount);
  for (int b=0; b<batchCount; b++) {
    Dense A = vecA[b];
    for (int i=0; i<A.dim[0]; i++) {
      for (int j=0; j<A.dim[1]; j++) {
        h_A[i+j*A.dim[0]+b*max_m*max_k] = A(i,j);
      }
    }
    Dense B = vecB[b];
    for (int i=0; i<B.dim[0]; i++) {
      for (int j=0; j<B.dim[1]; j++) {
        h_B[i+j*B.dim[0]+b*max_k*max_n] = B(i,j);
      }
    }
    Dense C = vecC[b];
    for (int i=0; i<C.dim[0]; i++) {
      for (int j=0; j<C.dim[1]; j++) {
        h_C[i+j*C.dim[0]+b*max_m*max_n] = C(i,j);
      }
    }
  }
  kblasHandle_t handle;
  kblasCreate(&handle);
  kblasEnableMagma(handle);
  magma_init();
  int *d_m, *d_n, *d_k;
  cudaMalloc( (void**)&d_m, batchCount * sizeof(int) );
  cudaMalloc( (void**)&d_n, batchCount * sizeof(int) );
  cudaMalloc( (void**)&d_k, batchCount * sizeof(int) );
  double *d_A, *d_B, *d_C;
  cudaMalloc( (void**)&d_A, h_A.size() * sizeof(double) );
  cudaMalloc( (void**)&d_B, h_B.size() * sizeof(double) );
  cudaMalloc( (void**)&d_C, h_C.size() * sizeof(double) );
  double **p_A, **p_B, **p_C;
  cudaMalloc( (void**)&p_A, batchCount * sizeof(double*) );
  cudaMalloc( (void**)&p_B, batchCount * sizeof(double*) );
  cudaMalloc( (void**)&p_C, batchCount * sizeof(double*) );
  generateDArrayOfPointers(d_A, p_A, max_m * max_k, batchCount, 0);
  generateDArrayOfPointers(d_B, p_B, max_k * max_n, batchCount, 0);
  generateDArrayOfPointers(d_C, p_C, max_m * max_n, batchCount, 0);
  cudaMemcpy(d_m, &h_m[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, &h_n[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, &h_k[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, &h_A[0], h_A.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, &h_B[0], h_B.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, &h_C[0], h_C.size() * sizeof(double), cudaMemcpyHostToDevice);
  kblas_gemm_batch_nonuniform_wsquery(handle);
  kblasAllocateWorkspace(handle);
  start("Batched GEMM");
  kblas_gemm_batch(handle, 'N', 'N', d_m, d_n, d_k, max_m, max_n, max_k,
                   alpha, (const double**)p_A, d_m, (const double**)p_B, d_k,
                   beta, p_C, d_m, batchCount );
  stop("Batched GEMM");
  cudaMemcpy(&h_C[0], d_C, max_m * max_n * batchCount * sizeof(double), cudaMemcpyDeviceToHost);
  for (int b=0; b<batchCount; b++) {
    Dense C(h_m[b],h_n[b]);
    for (int i=0; i<C.dim[0]; i++) {
      for (int j=0; j<C.dim[1]; j++) {
        C(i,j) = h_C[i+j*C.dim[0]+b*max_m*max_n];
      }
    }
    Dense D = vecC[b];
    D.gemm(vecA[b], vecB[b], alpha, beta);
    double diff = (C - D).norm();
    double norm = D.norm();
    print("Rel. L2 Error", std::sqrt(diff/norm), false);
  }
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_m);
  cudaFree(d_n);
  cudaFree(d_k);
  cudaFree(p_A);
  cudaFree(p_B);
  cudaFree(p_C);
  kblasDestroy(&handle);
  magma_finalize();
  return 0;
}
