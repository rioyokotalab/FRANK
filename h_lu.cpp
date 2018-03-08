#include "mpi_utils.h"
#include "functions.h"
#include "hblas.h"
#include "print.h"
#include "timer.h"

using namespace hicma;

#define BLOCK_LU 1

int main(int argc, char** argv) {
  int N = 64;
  int rank = 8;
  std::vector<double> randx(N);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  print("Time");
  start("Init matrix");
#if BLOCK_LU
  int nleaf = 16; // 4 x 4 leafs
  int nblocks = N / nleaf; // 1 level
  int admis = nblocks; // Full rank
#elif HODFR
#elif BLR
#elif H-LU
#endif
  Hierarchical A(laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical x(rand, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  Hierarchical b(zeros, randx, N, 1, rank, nleaf, admis, nblocks, 1);
  b -= A * x;
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
