#include "low_rank.h"
#include "functions.h"
#include "print.h"
#include "timer.h"

#include <algorithm>
#include <cublas_v2.h>
#include <magma_v2.h>
#include "kblas.h"
#include "testing_helper.h"
#include "batch_rand.h"
#include "batch_ara.h"

#include <iostream>
#include <fstream>

using namespace hicma;

int main(int argc, char** argv)
{
  int batchCount = 4;
  double tol = 1e-7;
  int block_size = 32;
  int ara_r = 10;
  std::vector<Dense> vecA;
#if 1
  for (int b=0; b<batchCount; b++) {
    int m = (b+1)*20;
    int n = (b+1)*16;
    m = 16;
    n = 16;
    std::vector<double> randx(2*m);
    for (int i=0; i<2*m; i++) {
      randx[i] = drand48();
    }
    std::sort(randx.begin(), randx.end());
    Dense A(laplace1d, randx, m, n, 0, n);
    vecA.push_back(A);
  }
#else
  int dim0, dim1;
  std::ifstream file("matrix.txt");
  file >> batchCount;
  for (int b=0; b<batchCount; b++) {
    file >> dim0;
    file >> dim1;
    std::cout << b << " " << dim0 << " " << dim1 << std::endl;
    Dense A(dim0,dim1);
    for (int i=0; i<dim0; i++) {
      for (int j=0; j<dim1; j++) {
        double Aij;
        file >> Aij;
        A(i,j) = Aij;
      }
    }
    vecA.push_back(A);
  }
  file.close();
#endif
  std::vector<int> h_m(batchCount);
  std::vector<int> h_n(batchCount);
  std::vector<int> h_ldm(batchCount);
  std::vector<int> h_ldn(batchCount);
  int max_m=0, max_n=0;
  for (int b=0; b<batchCount; b++) {
    Dense A = vecA[b];
    h_m[b] = A.dim[0];
    h_n[b] = A.dim[1];
    h_ldm[b] = std::max(h_m[b],32);
    h_ldn[b] = std::max(h_n[b],32);
    max_m = std::max(max_m, h_ldm[b]);
    max_n = std::max(max_n, h_ldn[b]);
    printf("%d %d %d\n",b,h_m[b],h_n[b]);
  }
  std::vector<double> h_A(max_m * max_n * batchCount);
  std::vector<double> h_U(max_m * max_n * batchCount);
  std::vector<double> h_V(max_n * max_n * batchCount);
  for (int b=0; b<batchCount; b++) {
    Dense A = vecA[b];
    for (int i=0; i<A.dim[0]; i++) {
      for (int j=0; j<A.dim[1]; j++) {
        h_A[i+j*h_ldm[b]+b*max_m*max_n] = A(i,j);
      }
    }
  }
  kblasHandle_t handle;
  kblasRandState_t rand_state;
  kblasCreate(&handle);
  kblasInitRandState(handle, &rand_state, 16384*2, 0);
  kblasEnableMagma(handle);
  magma_init();
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
  generateDArrayOfPointers(d_A, p_A, max_m * max_n, batchCount, 0);
  generateDArrayOfPointers(d_U, p_U, max_m * max_n, batchCount, 0);
  generateDArrayOfPointers(d_V, p_V, max_n * max_n, batchCount, 0);
  kblas_ara_batch_wsquery<double>(handle, block_size, batchCount);
  kblasAllocateWorkspace(handle);
  cudaMemcpy(d_m, &h_m[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, &h_n[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ldm, &h_ldm[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ldn, &h_ldn[0], batchCount * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, &h_A[0], h_A.size() * sizeof(double), cudaMemcpyHostToDevice);
  start("Batched RSVD");
  kblas_ara_batched(handle, d_m, d_n, p_A, d_ldm, p_U, d_ldm, p_V, d_ldn, d_k,
                    tol, max_m, max_n, max_n, block_size, ara_r, rand_state, batchCount);
  stop("Batched RSVD");
  std::vector<int> h_k(batchCount);
  cudaMemcpy(&h_k[0], d_k, batchCount * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_U[0], d_U, h_U.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_V[0], d_V, h_V.size() * sizeof(double), cudaMemcpyDeviceToHost);
  std::vector<LowRank> vecLR;
  for(int b=0; b<batchCount; b++) {
    LowRank LR(vecA[b].dim[0], vecA[b].dim[1], h_k[b]);
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
    vecLR.push_back(LR);
  }
  for (int b=0; b<batchCount; b++) {
    double diff = (vecA[b] - Dense(vecLR[b])).norm();
    double norm = vecA[b].norm();
    print("rank", h_k[b]);
    print("Rel. L2 Error", std::sqrt(diff/norm), false);
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
  cudaFree(d_ldm);
  cudaFree(d_ldn);
  return 0;
}
