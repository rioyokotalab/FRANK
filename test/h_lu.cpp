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
  timing::start("Init matrix");
  timing::start("CPU compression");
  Hierarchical A(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  timing::stop("CPU compression");
  rsvd_batch();
  // printXML(A);
  admis = N / nleaf; // Full rank
  Dense x(random_uniform, randx, N, 1);
  Dense b(N, 1);
  // timing::start("Dense tree");
  // Hierarchical D(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  // timing::stop("Dense tree");
  // timing::start("Verification time");
  // print("Compression Accuracy");
  // print("Rel. L2 Error", l2_error(A, D), false);
  // timing::stop("Verification time");
  print("Time");
  gemm(A, x, b, 1, 1);
  gemm_batch();
  timing::stopAndPrint("Init matrix");
  timing::start("LU decomposition");
  Hierarchical L, U;
  std::tie(L, U) = getrf(A);
  timing::stopAndPrint("LU decomposition", 2);
  timing::start("Verification time");
  timing::start("Forward substitution");
  trsm(L, b,'l');
  timing::stop("Forward substitution");
  timing::start("Backward substitution");
  trsm(U, b,'u');
  timing::stop("Backward substitution");
  print("LU Accuracy");
  print("Rel. L2 Error", l2_error(x, b), false);
  timing::stopAndPrint("Verification time");
  return 0;
}
