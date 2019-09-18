#include "any.h"
#include "low_rank.h"
#include "hierarchical.h"
#include "functions.h"
#include "batch.h"
#include "print.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 1 << atoi(argv[2]);
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
  //printXML(A);
  admis = N / nleaf; // Full rank
  start("Dense tree");
  //Hierarchical D(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  stop("Dense tree");
  Hierarchical x(random, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  Hierarchical b(zeros, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  //start("Verification");
  //double diff = (Dense(A) - Dense(D)).norm();
  //double norm = D.norm();
  //stop("Verification");
  //print("Compression Accuracy");
  //print("Rel. L2 Error", std::sqrt(diff/norm), false);
  print("Time");
  b.gemm(A, x, 1, 1);
  gemm_batch();
  stop("Init matrix");
  printTime("-DGEMM");
  start("LU decomposition");
  A.getrf();
  stop("LU decomposition");
  printTime("-DGETRF");
  printTime("-DTRSM");
  printTime("-DGEMM");
  start("Forward substitution");
  b.trsm(A,'l');
  stop("Forward substitution");
  printTime("-DTRSM");
  printTime("-DGEMM");
  start("Backward substitution");
  b.trsm(A,'u');
  stop("Backward substitution");
  printTime("-DTRSM");
  printTime("-DGEMM");
  double diff = (Dense(x) - Dense(b)).norm();
  double norm = x.norm();
  print("LU Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
