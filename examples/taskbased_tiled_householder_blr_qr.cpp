#include "hicma/hicma.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <omp.h>
#include <sys/time.h>
#include <cstdlib>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

using namespace hicma;

int main(int argc, char** argv) {
  hicma::initialize();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t Nb = argc > 2 ? atoi(argv[2]) : 32;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 16;
  double admis = argc > 4 ? atof(argv[4]) : 0;
  int64_t Nc = N / Nb;
  setGlobalValue("HICMA_LRA", "rounded_addition");
  setGlobalValue("HICMA_DISABLE_TIMER", "1");

  std::vector<std::vector<double>> randpts;
  randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  Hierarchical D(laplacend, randpts, N, N, Nb, Nb, Nc, Nc, Nc);
  Hierarchical A(laplacend, randpts, N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical A_copy(A);
  print("BLR Compression Accuracy");
  print("Rel. L2 Error", l2_error(D, A), false);

  Hierarchical Q(identity, std::vector<std::vector<double>>(), N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical T(Nc, Nc);
  for(int64_t j = 0; j < Nc; j++) {
    for(int64_t i = j; i < Nc; i++) {
      T(i, j) = Dense(get_n_cols(A(j, j)), get_n_cols(A(j, j)));
    }
  }

  print("Taskbased Tiled Householder BLR-QR");
  print("Time");
  double tic = get_time();
  char dep[Nc][Nc];
  #pragma omp parallel
  {
    #pragma omp single
    {
      for(int64_t k = 0; k < Nc; k++) {
        #pragma omp task depend(out: dep[k][k]) priority(3)
        {
          geqrt(A(k, k), T(k, k));
        }
        for(int64_t j = k+1; j < Nc; j++) {
          #pragma omp task depend(in: dep[k][k]) depend(out: dep[k][j]) priority(0)
          {
            larfb(A(k, k), T(k, k), A(k, j), true);
          }
        }
        for(int64_t i = k+1; i < Nc; i++) {
          #pragma omp task depend(in: dep[i-1][k]) depend(out: dep[i][k]) priority(2)
          {
            tpqrt(A(k, k), A(i, k), T(i, k));
          }
          for(int64_t j = k+1; j < Nc; j++) {
            #pragma omp task depend(in: dep[i][k], dep[i-1][j]) depend(out: dep[i][j]) priority(1)
            {
              tpmqrt(A(i, k), T(i, k), A(k, j), A(i, j), true);
            }
          }
        }
      }
    }
  }
  double toc = get_time();
  print("BLR-QR", toc-tic);
  //Build Q: Apply Q to Id
  #pragma omp parallel
  {
    #pragma omp single
    {
      for(int64_t k = Nc-1; k >= 0; k--) {
        for(int64_t i = Nc-1; i > k; i--) {
          for(int64_t j = k; j < Nc; j++) {
            #pragma omp task depend(inout: dep[k][j], dep[i][j])
            {
              tpmqrt(A(i, k), T(i, k), Q(k, j), Q(i, j), false);
            }
          }
        }
        for(int64_t j = k; j < Nc; j++) {
          #pragma omp task depend(inout: dep[k][j])
          {
            larfb(A(k, k), T(k, k), Q(k, j), false);
          }
        }
      }
    }
  }
  //Build R: Take upper triangular part of modified A
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<=i; j++) {
      if(i == j) //Diagonal must be dense, zero lower-triangular part
        zero_lowtri(A(i, j));
      else
        zero_whole(A(i, j));
    }
  }
  
  print("BLR-QR Accuracy");
  //Residual
  Hierarchical QR(zeros, std::vector<std::vector<double>>(), N, N, rank, Nb, admis, Nc, Nc);
  gemm(Q, A, QR, 1, 0);
  print("Residual", l2_error(A_copy, QR), false);  
  //Orthogonality
  Hierarchical QtQ(zeros, std::vector<std::vector<double>>(), N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical Qt = transpose(Q);
  gemm(Qt, Q, QtQ, 1, 0);
  print("Orthogonality", l2_error(Dense(identity, std::vector<std::vector<double>>(), N, N), QtQ), false);
  return 0;
}
