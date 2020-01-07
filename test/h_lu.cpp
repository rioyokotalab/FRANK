#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main(int argc, char** argv) {
  yorel::multi_methods::initialize();
  int N = argc > 1 ? atoi(argv[1]) : 256;
  int nleaf = argc > 2 ? atoi(argv[2]) : 16;
  int rank = argc > 3 ? atoi(argv[3]) : 8;
  int nblocks = argc > 4 ? atoi(argv[4]) : 2;
  int admis = argc > 5 ? atoi(argv[5]) : 0;
  std::vector<double> randx(N);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  start("Init matrix");
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
