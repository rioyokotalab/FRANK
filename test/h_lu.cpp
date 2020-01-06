#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main(int argc, char** argv) {
  yorel::multi_methods::initialize();
  int N = atoi(argv[2]);
  int nleaf = atoi(argv[3]);
  int rank = atoi(argv[4]);
  std::vector<double> randx(N);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  start("Init matrix");
  int nblocks=0, admis=0;
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
  start("CPU compression");
  Hierarchical A(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  stop("CPU compression");
  rsvd_batch();
  // printXML(A);
  admis = N / nleaf; // Full rank
  Hierarchical x(random_uniform, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  Hierarchical b(zeros, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  // start("Dense tree");
  // Hierarchical D(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  // stop("Dense tree");
  // start("Verification time");
  // print("Compression Accuracy");
  // print("Rel. L2 Error", l2_error(A, D), false);
  // stop("Verification time");
  print("Time");
  gemm(A, x, b, 1, 1);
  gemm_batch();
  stop("Init matrix");
  printTime("-DGEMM");
  start("LU decomposition");
  Hierarchical L, U;
  std::tie(L, U) = getrf(A);
  stop("LU decomposition");
  printTime("-DGETRF");
  printTime("-DTRSM");
  printTime("-DGEMM");
  start("Forward substitution");
  trsm(L, b,'l');
  stop("Forward substitution");
  printTime("-DTRSM");
  printTime("-DGEMM");
  start("Backward substitution");
  trsm(U, b,'u');
  stop("Backward substitution");
  printTime("-DTRSM");
  printTime("-DGEMM");
  print("LU Accuracy");
  print("Rel. L2 Error", l2_error(x, b), false);
  return 0;
}
