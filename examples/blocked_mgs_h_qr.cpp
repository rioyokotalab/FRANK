#include "hicma/hicma.h"

#include <cstdint>
#include <iostream>
#include <vector>


using namespace hicma;

int main(int argc, char** argv) {
  hicma::initialize();
  hicma::setGlobalValue("HICMA_LRA", "rounded_addition");
  constexpr int64_t N = 128;
  constexpr int64_t nleaf = 16;
  constexpr double eps = 1e-6;
  const std::vector<std::vector<double>> randx{
    get_sorted_random_vector(N)
  };
  timing::start("Init matrix");
  int64_t nblocks=0;
  double admis=0;
  if(argc < 2) {
    std::cout <<"Argument(s) needed" <<std::endl;
    exit(1);
  }
  if (atoi(argv[1]) == 0) {
    nblocks = N / nleaf; // 1 level
    admis = N / nleaf; // Full rank
  }
  else if (atoi(argv[1]) == 1) {
    nblocks = 2; // Hierarchical (log_2(N/nleaf) levels)
    admis = N / nleaf; // Full rank
  }
  else if (atoi(argv[1]) == 2) {
    nblocks = N / nleaf; // 1 level
    admis = 0; // Weak admissibility
  }
  else if (atoi(argv[1]) == 3) {
    nblocks = N / nleaf; // 1 level
    admis = 1; // Strong admissibility
  }
  else if (atoi(argv[1]) == 4) {
    nblocks = 2; // Hierarchical (log_2(N/nleaf) levels)
    admis = 0; // Weak admissibility
  }
  else {
    nblocks = 2; // Hierarchical (log_2(N/nleaf) levels)
    admis = 1; // Strong admissibility
  }
  timing::start("Dense tree");
  const Hierarchical D(laplacend, randx, N, N, nleaf, nleaf, N / nleaf, nblocks, nblocks);
  timing::stopAndPrint("Dense tree");
  timing::start("CPU compression");
  Hierarchical A(laplacend, randx, N, N, nleaf, eps, admis, nblocks, nblocks);
  timing::stop("CPU compression");
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(D, A), false);

  Hierarchical Q(A);
  Hierarchical R(A);
  print("Blocked Modified Gram-Schmidt H-QR");
  print("Time");
  timing::start("H-QR");
  mgs_qr(A, Q, R);
  timing::stopAndPrint("H-QR", 1);
  
  print("H-QR Accuracy");
  Hierarchical QR(Q);
  trmm(R, QR, hicma::Side::Right, hicma::Mode::Upper, 'n', 'n', 1.);
  print("Residual", l2_error(D, QR), false);
  
  Hierarchical QtQ(zeros, randx, N, N, nleaf, eps, admis, nblocks, nblocks);
  const Hierarchical Qt = transpose(Q);
  gemm(Qt, Q, QtQ, 1, 0);
  print("Orthogonality", l2_error(Dense(identity, randx, N, N), QtQ), false);
  return 0;
}
