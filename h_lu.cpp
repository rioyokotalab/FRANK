#include "mpi_utils.h"
#include "functions.h"
#include "hblas.h"
#include "print.h"
#include "timer.h"

using namespace hicma;

int main(int argc, char** argv) {
  int N = 64;
  int Nb = 16;
  int Nc = N / Nb;
  int rank = 8;
  int admis = Nc;
  std::vector<double> randx(N);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  print("Time");
  start("Init matrix");
  Hierarchical A(laplace1d, randx, N, N, rank, Nb, admis, Nc, Nc, 0, 0, 0, 0, 0);
  Hierarchical x(rand, randx, N, 1, rank, Nb, admis, Nc, 1, 0, 0, 0, 0, 0);
  Hierarchical b(zeros, randx, N, 1, rank, Nb, admis, Nc, 1, 0, 0, 0, 0, 0);
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
