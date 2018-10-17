#include "any.h"
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
  int batchCount = 2;
  double tol = 1e-7;
  int block_size = 32;
  int ara_r = 10;
  std::vector<int> h_m;
  std::vector<int> h_n;
  std::vector<Dense> vecA;
#if 1
  for (int b=0; b<batchCount; b++) {
    int N = 64;
    if(b==1) N = 128;
    std::vector<double> randx(2*N);
    for (int i=0; i<2*N; i++) {
      randx[i] = drand48();
    }
    std::sort(randx.begin(), randx.end());
    Dense A(laplace1d, randx, N, N-2, 0, N);
    h_m.push_back(A.dim[0]);
    h_n.push_back(A.dim[1]);
    vecA.push_back(A);
  }
#else
  int dim0, dim1;
  std::ifstream file("matrix.txt");
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
    h_m.push_back(A.dim[0]);
    h_n.push_back(A.dim[1]);
    vecA.push_back(A);
  }
  file.close();
#endif
  int max_m = 0;
  int max_n = 0;
  for (int b=0; b<batchCount; b++) {
    max_m = std::max(max_m, h_m[b]);
    max_n = std::max(max_n, h_n[b]);
  }
  std::vector<double> h_A(max_m * max_n * batchCount);
  std::vector<double> h_U(max_m * max_n * batchCount);
  std::vector<double> h_V(max_n * max_n * batchCount);
  for (int b=0; b<batchCount; b++) {
    Dense A = vecA[b];
    for (int i=0; i<A.dim[0]; i++) {
      for (int j=0; j<A.dim[1]; j++) {
        h_A[i+j*A.dim[0]+b*max_m*max_n] = A(i,j);
      }
    }
  }
  kblasHandle_t handle;
  kblasRandState_t rand_state;
  kblasCreate(&handle);
  kblasInitRandState(handle, &rand_state, 16384*2, 0);
  kblasEnableMagma(handle);
  magma_init();
  int *d_m, *d_n, *d_k;
  cudaMalloc( (void**)&d_m, batchCount * sizeof(int) );
  cudaMalloc( (void**)&d_n, batchCount * sizeof(int) );
  cudaMalloc( (void**)&d_k, batchCount * sizeof(int) );
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
  cudaMemcpy(d_A, &h_A[0], h_A.size() * sizeof(double), cudaMemcpyHostToDevice);
  start("Batched RSVD");
  kblas_ara_batched(
                    handle, d_m, d_n, p_A, d_m, p_U, d_m, p_V, d_n, d_k,
                    tol, max_m, max_n, max_n, block_size, ara_r, rand_state, batchCount
                    );
  stop("Batched RSVD");
  std::vector<int> h_k(batchCount);
  cudaMemcpy(&h_k[0], d_k, batchCount * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_U[0], d_U, h_U.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_V[0], d_V, h_V.size() * sizeof(double), cudaMemcpyDeviceToHost);
  std::vector<LowRank> vecLR;
  for(int b=0; b<batchCount; b++) {
    std::cout << b << " " << h_k[b] << std::endl;
    LowRank LR(vecA[b].dim[0], vecA[b].dim[1], h_k[b]);
    Dense A = vecA[b];
    for (int i=0; i<LR.dim[0]; i++) {
      for (int j=0; j<LR.rank; j++) {
        LR.U(i,j) = h_U[i+j*LR.dim[0]+b*max_m*max_n];
      }
    }
    for (int i=0; i<LR.rank; i++) {
      for (int j=0; j<LR.dim[1]; j++) {
        LR.V(i,j) = h_V[i*LR.dim[1]+j+b*max_n*max_n];
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
  return 0;
}
