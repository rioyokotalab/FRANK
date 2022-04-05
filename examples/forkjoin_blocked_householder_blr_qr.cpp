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

  Hierarchical Q(identity, {}, N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical T(Nc, 1);
  print("Blocked Householder BLR-QR");
  print("Time");
  double tic = get_time();
  for(int64_t k = 0; k < Nc; k++) {
    triangularize_block_col(k, A, T);
    #pragma omp parallel for schedule(dynamic)
    for(int j = k+1; j < Nc; j++) {
      apply_block_col_householder(A, T, k, true, A, j);
    }
  }
  double toc = get_time();
  print("BLR-QR", toc-tic);
  //Build Q: Apply Q to Id
  for(int64_t k = Nc-1; k >= 0; k--) {
    #pragma omp parallel for schedule(dynamic)
    for(int j = k; j < Nc; j++) {
      apply_block_col_householder(A, T, k, false, Q, j);
    }
  }
  //Build R: Take upper triangular part of modified A
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<=i; j++) {
      if(i == j) //Diagonal must be dense, zero lower-triangular part
        zero_lower(A(i, j));
      else
        zero_all(A(i, j));
    }
  }

  print("BLR-QR Accuracy");
  //Residual
  Hierarchical QR(zeros, {}, N, N, rank, Nb, admis, Nc, Nc);
  gemm(Q, A, QR, 1, 0);
  print("Residual", l2_error(A_copy, QR), false);  
  //Orthogonality
  Hierarchical QtQ(zeros, {}, N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical Qt = transpose(Q);
  gemm(Qt, Q, QtQ, 1, 0);
  print("Orthogonality", l2_error(Dense(identity, {}, N, N), QtQ), false);
  return 0;
}
