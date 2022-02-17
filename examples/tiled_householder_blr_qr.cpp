#include "hicma/hicma.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>


using namespace hicma;

int main(int argc, char** argv) {
  hicma::initialize();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t Nb = argc > 2 ? atoi(argv[2]) : 32;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 16;
  double admis = argc > 4 ? atof(argv[4]) : 0;
  int64_t Nc = N / Nb;
  setGlobalValue("HICMA_LRA", "rounded_orth");

  std::vector<std::vector<double>> randpts;
  randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  Hierarchical<double> D(LaplacendKernel<double>(randpts), N, N, Nb, Nb, Nc, Nc, Nc);
  Hierarchical<double> A(LaplacendKernel<double>(randpts), N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical A_copy(A);
  print("BLR Compression Accuracy");
  print("Rel. L2 Error", l2_error(D, A), false);

  Hierarchical<double> Q(IdentityKernel<double>(), N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical T(Nc, Nc);
  for(int64_t j = 0; j < Nc; j++) {
    for(int64_t i = j; i < Nc; i++) {
      T(i, j) = Dense(get_n_cols(A(j, j)), get_n_cols(A(j, j)));
    }
  }
  print("Tiled Householder BLR-QR");
  print("Time");
  timing::start("BLR-QR");
  for(int64_t k = 0; k < Nc; k++) {
    geqrt(A(k, k), T(k, k));
    for(int64_t j = k+1; j < Nc; j++) {
      larfb(A(k, k), T(k, k), A(k, j), true);
    }
    for(int64_t i = k+1; i < Nc; i++) {
      tpqrt(A(k, k), A(i, k), T(i, k));
      for(int64_t j = k+1; j < Nc; j++) {
        tpmqrt(A(i, k), T(i, k), A(k, j), A(i, j), true);
      }
    }
  }
  timing::stopAndPrint("BLR-QR", 1);
  //Build Q: Apply Q to Id
  for(int64_t k = Nc-1; k >= 0; k--) {
    for(int64_t i = Nc-1; i > k; i--) {
      for(int64_t j = k; j < Nc; j++) {
        tpmqrt(A(i, k), T(i, k), Q(k, j), Q(i, j), false);
      }
    }
    for(int64_t j = k; j < Nc; j++) {
      larfb(A(k, k), T(k, k), Q(k, j), false);
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
  Hierarchical<double> QR(ZeroKernel<double>(), N, N, rank, Nb, admis, Nc, Nc);
  gemm(Q, A, QR, 1, 0);
  print("Residual", l2_error(A_copy, QR), false);  
  //Orthogonality
  Hierarchical<double> QtQ(ZeroKernel<double>(), N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical Qt = transpose(Q);
  gemm(Qt, Q, QtQ, 1, 0);
  print("Orthogonality", l2_error(Dense<double>(IdentityKernel<double>(), N, N), QtQ), false);
  return 0;
}
