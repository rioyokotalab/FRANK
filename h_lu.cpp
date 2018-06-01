#include "mpi_utils.h"
#include "functions.h"
#include "hblas.h"
#include "print.h"
#include "timer.h"

using namespace hicma;

int main(int argc, char** argv) {
  int N = 64;
  int nleaf = 16;
  int rank = 8;
  std::vector<double> randx(N);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  print("Time");
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
    admis = 1; // Weak admissibility
  }
  else if (atoi(argv[1]) == 4) {
    nblocks = N / nleaf; // 1 level
    admis = 2; // Strong admissibility
  }
  Hierarchical A(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical A_test(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical x(rand, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  Hierarchical b(zeros, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  Hierarchical b_test(zeros, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  b -= A * x;
  b_test += A_test.mul(x);
  stop("Init matrix");
  start("LU decomposition");
  A.getrf();
  A_test.getrf_test();
  stop("LU decomposition");
  start("Forward substitution");
  b.trsm(A,'l');
  b_test.trsm_test(A_test,'l');
  stop("Forward substitution");
  start("Backward substitution");
  b.trsm(A,'u');
  b_test.trsm_test(A_test,'u');
  stop("Backward substitution");
  double diff = (x - b).norm();
  double diff_test = (*x.sub(b_test)).norm_test();
  double norm = x.norm();
  double norm_test = x.norm_test();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  print("Rel. L2 Error", std::sqrt(diff_test/norm_test), false);
  return 0;
}
