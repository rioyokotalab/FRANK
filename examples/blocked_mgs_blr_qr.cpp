#include "hicma/hicma.h"

#include <algorithm>
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

  Hierarchical<double> Q(ZeroKernel<double>(), N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical<double> R(ZeroKernel<double>(), N, N, rank, Nb, admis, Nc, Nc);
  print("Blocked Modified Gram-Schmidt BLR-QR");
  print("Time");
  timing::start("BLR-QR");
  qr(A, Q, R);
  timing::stopAndPrint("BLR-QR", 1);

  print("BLR-QR Accuracy");
  //Residual
  Hierarchical<double> QR(ZeroKernel<double>(), N, N, rank, Nb, admis, Nc, Nc);
  gemm(Q, R, QR, 1, 0);
  print("Residual", l2_error(A_copy, QR), false);
  //Orthogonality
  Hierarchical<double> QtQ(ZeroKernel<double>(), N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical Qt = transpose(Q);
  gemm(Qt, Q, QtQ, 1, 0);
  print("Orthogonality", l2_error(Dense(identity, std::vector<std::vector<double>>(), N, N), QtQ), false);
  return 0;
}
