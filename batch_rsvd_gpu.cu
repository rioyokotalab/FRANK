#include "any.h"
#include "low_rank.h"
#include "batch_rsvd.h"
#include "timer.h"

#include <cublas_v2.h>
#include <magma_v2.h>
#include "kblas.h"
#include "testing_helper.h"
#include "batch_rand.h"
#include "batch_ara.h"

#include <iostream>
#include <fstream>

namespace hicma {

  std::vector<Dense> vecA;
  std::vector<Any*> vecLR;

  void low_rank_push(Any& A, Dense& Aij, int rank) {
    vecA.push_back(Aij);
    vecLR.push_back(&A);
  }

  void batch_rsvd() {
    int batchCount = vecA.size();
    if (batchCount == 0) return;
    double tol = 1e-7;
    int block_size = 32;
    int ara_r = 10;
    int max_m = 0;
    int max_n = 0;
    std::vector<int> h_m(batchCount);
    std::vector<int> h_n(batchCount);
    std::vector<int> h_ldm(batchCount);
    std::vector<int> h_ldn(batchCount);
    for (int b=0; b<batchCount; b++) {
      Dense A = vecA[b];
      h_m[b] = A.dim[0];
      h_n[b] = A.dim[1];
      h_ldm[b] = std::max(h_m[b],32);
      h_ldn[b] = std::max(h_n[b],32);
      max_m = std::max(max_m, h_m[b]);
      max_n = std::max(max_n, h_n[b]);
    }
    start("Allocate host");
    std::vector<double> h_A(max_m * max_n * batchCount);
    std::vector<double> h_U(max_m * max_n * batchCount);
    std::vector<double> h_V(max_n * max_n * batchCount);
    stop("Allocate host");
    start("Copy matrix");
    for (int b=0; b<batchCount; b++) {
      Dense A = vecA[b];
      for (int i=0; i<A.dim[0]; i++) {
        for (int j=0; j<A.dim[1]; j++) {
          h_A[i+j*h_ldm[b]+b*max_m*max_n] = A(i,j);
        }
      }
    }
    stop("Copy matrix");
#if 0
    start("Write matrix");
    std::ofstream file("matrix.txt");
    file << batchCount << std::endl;
    for (int b=0; b<batchCount; b++) {
      Dense A = vecA[b];
      file << A.dim[0] << std::endl;
      file << A.dim[1] << std::endl;
      for (int i=0; i<A.dim[0]; i++) {
        for (int j=0; j<A.dim[1]; j++) {
          file << A(i,j) << std::endl;
        }
      }
    }
    stop("Write matrix");
#endif
    start("Init KBLAS");
    kblasHandle_t handle;
    kblasRandState_t rand_state;
    kblasCreate(&handle);
    kblasInitRandState(handle, &rand_state, 16384*2, 0);
    kblasEnableMagma(handle);
    magma_init();
    stop("Init KBLAS");
    start("Allocate device");
    int *d_m, *d_n, *d_k, *d_ldm, *d_ldn;
    cudaMalloc( (void**)&d_m, batchCount * sizeof(int) );
    cudaMalloc( (void**)&d_n, batchCount * sizeof(int) );
    cudaMalloc( (void**)&d_k, batchCount * sizeof(int) );
    cudaMalloc( (void**)&d_ldm, batchCount * sizeof(int) );
    cudaMalloc( (void**)&d_ldn, batchCount * sizeof(int) );
    double *d_A, *d_U, *d_V;
    cudaMalloc( (void**)&d_A, h_A.size() * sizeof(double) );
    cudaMalloc( (void**)&d_U, h_U.size() * sizeof(double) );
    cudaMalloc( (void**)&d_V, h_V.size() * sizeof(double) );
    double **p_A, **p_U, **p_V;
    cudaMalloc( (void**)&p_A, batchCount * sizeof(double*) );
    cudaMalloc( (void**)&p_U, batchCount * sizeof(double*) );
    cudaMalloc( (void**)&p_V, batchCount * sizeof(double*) );
    stop("Allocate device");
    start("Array of pointers");
    generateDArrayOfPointers(d_A, p_A, max_m * max_n, batchCount, 0);
    generateDArrayOfPointers(d_U, p_U, max_m * max_n, batchCount, 0);
    generateDArrayOfPointers(d_V, p_V, max_n * max_n, batchCount, 0);
    stop("Array of pointers");
    start("Copy to device");
    kblas_ara_batch_wsquery<double>(handle, block_size, batchCount);
    kblasAllocateWorkspace(handle);
    cudaMemcpy(d_m, &h_m[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &h_n[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldm, &h_ldm[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldn, &h_ldn[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, &h_A[0], h_A.size() * sizeof(double), cudaMemcpyHostToDevice);
    stop("Copy to device");
    start("Batched SVD");
    kblas_ara_batched(handle, d_m, d_n, p_A, d_ldm, p_U, d_ldm, p_V, d_ldn, d_k,
                      tol, max_m, max_n, max_n, block_size, ara_r, rand_state, batchCount);
    cudaDeviceSynchronize();
    stop("Batched SVD");
    start("Copy to host");
    std::vector<int> h_k(batchCount);
    cudaMemcpy(&h_k[0], d_k, batchCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_U[0], d_U, h_U.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_V[0], d_V, h_V.size() * sizeof(double), cudaMemcpyDeviceToHost);
    stop("Copy to host");
    start("Copy to LR");
    for(int b=0; b<batchCount; b++) {
      assert(h_k[b] != 0);
      LowRank LR(vecA[b].dim[0], vecA[b].dim[1], h_k[b]);
      //std::cout << LR.dim[0] << " " << LR.dim[1] << " " << LR.rank << std::endl;
      Dense A = vecA[b];
      for (int i=0; i<LR.dim[0]; i++) {
        for (int j=0; j<LR.rank; j++) {
          LR.U(i,j) = h_U[i+j*h_ldm[b]+b*max_m*max_n];
        }
      }
      for (int i=0; i<LR.rank; i++) {
        for (int j=0; j<LR.dim[1]; j++) {
          LR.V(i,j) = h_V[i*h_ldn[b]+j+b*max_n*max_n];
        }
        LR.S(i,i) = 1;
      }
      *vecLR[b] = LR;
    }
    stop("Copy to LR");
    cudaFree(p_A);
    cudaFree(p_U);
    cudaFree(p_V);
    cudaFree(d_A);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_k);
    cudaFree(d_ldm);
    cudaFree(d_ldn);
    kblasDestroy(&handle);
  }
}
