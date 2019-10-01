#include "hicma/any.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 128;
  int nleaf = 16;
  int rank = 8;
  std::vector<double> randx(N);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  start("Init matrix");
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
  start("CPU compression");
  Hierarchical A(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  stop("CPU compression");
  rsvd_batch();
  Hierarchical Q(zeros, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical R(zeros, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical QR(zeros, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  stop("Init matrix");
  printXML(A);
  admis = N / nleaf; // Full rank
  start("Dense tree");
  Hierarchical D(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  stop("Dense tree");
  start("Verification");
  double diff = (Dense(A) - Dense(D)).norm();
  double norm = D.norm();
  stop("Verification");
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  print("Time");
  start("QR decomposition");
  A.qr(Q, R);
  stop("QR decomposition");
  printTime("-DGEQRF");
  printTime("-DGEMM");
  QR.gemm(Q, R, 1, 1);
  diff = (Dense(QR) - Dense(D)).norm();
  norm = D.norm();
  print("QR Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  Dense DQ(Q);
  Dense QtQ(DQ.dim[1], DQ.dim[1]);
  QtQ.gemm(DQ, DQ, CblasTrans, CblasNoTrans, 1, 1);
  Dense Id(identity, randx, QtQ.dim[0], QtQ.dim[1]);
  diff = (QtQ - Id).norm();
  norm = Id.norm();
  print("Rel. L2 Orthogonality", std::sqrt(diff/norm), false);
  return 0;
}


