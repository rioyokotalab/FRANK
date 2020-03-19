#include "hicma/hicma.h"

#include "yorel/yomm2/cute.hpp"

#include <iostream>
#include <vector>


using namespace hicma;

int main(int argc, char** argv) {
  yorel::yomm2::update_methods();
  int N = 128;
  int nleaf = 16;
  int rank = 8;
  std::vector<double> randx = get_sorted_random_vector(N);
  timing::start("Init matrix");
  int nblocks=0, admis=0;
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
    nblocks = 4; // Hierarchical (log_4(N/nleaf) levels)
    admis = N / nleaf; // Full rank
  }
  else if (atoi(argv[1]) == 3) {
    nblocks = N / nleaf; // 1 level
    admis = 0; // Weak admissibility
  }
  else if (atoi(argv[1]) == 4) {
    nblocks = N / nleaf; // 1 level
    admis = 1; // Strong admissibility
  }
  else if (atoi(argv[1]) == 5) {
    nblocks = 2; // Hierarchical (log_2(N/nleaf) levels)
    admis = 0; // Weak admissibility
  }
  else if (atoi(argv[1]) == 6) {
    nblocks = 2; // Hierarchical (log_2(N/nleaf) levels)
    admis = 1; // Strong admissibility
  }
  timing::start("CPU compression");
  Hierarchical A(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  timing::stop("CPU compression");
  rsvd_batch();
  Hierarchical Q(zeros, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical R(zeros, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical QR(zeros, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  timing::stopAndPrint("Init matrix");
  admis = N / nleaf; // Full rank
  timing::start("Dense tree");
  Hierarchical D(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  timing::stopAndPrint("Dense tree");
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, D), false);
  print("Time");
  timing::start("QR decomposition");
  qr(A, Q, R);
  timing::stopAndPrint("QR decomposition", 1);
  gemm(Q, R, QR, 1, 1);
  print("QR Accuracy");
  print("Rel. L2 Error", l2_error(QR, D), false);
  Dense DQ(Q);
  Dense QtQ(DQ.dim[1], DQ.dim[1]);
  gemm(DQ, DQ, QtQ, true, false, 1, 1);
  Dense Id(identity, randx, QtQ.dim[0], QtQ.dim[1]);
  print("Rel. L2 Orthogonality", l2_error(QtQ, Id), false);
  return 0;
}
