#include "mpi_utils.h"
#include "functions.h"
#include "hblas.h"
#include "print.h"
#include "timer.h"

using namespace hicma;

#define BLOCK_LU 1
#define HODFR    0
#define BLR      0
#define H_LU     0

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
#if BLOCK_LU
  int nblocks = N / nleaf; // 1 level
  int admis = N / nleaf; // Full rank
#elif HODFR
  int nblocks = 2; // Hierarchical (log_2(N/nleaf) levels)
  int admis = N / nleaf; // Full rank
#elif BLR
#elif H_LU
#endif
  Hierarchical A(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical x(rand, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  Hierarchical b(zeros, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  b -= A * x;
#if BLOCK_LU
  D_t(b[0]).print();
#elif HODFR
  D_t(H_t(b[0])[0]).print();
#endif
  stop("Init matrix");
  start("LU decomposition");
  A.getrf();
  stop("LU decomposition");
  start("Forward substitution");
  b.trsm(A,'l');
  stop("Forward substitution");
  start("Backward substitution");
  b.trsm(A,'u');
  stop("Backward substitution");
  double diff = (x - b).norm();
  double norm = x.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
