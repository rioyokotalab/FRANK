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
  print2("-DGETRF");
  print2("-DTRSM");
  print2("-DGEMM");
  start("Forward substitution");
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<ic; jc++) {
      gemm(A(ic,jc),b[jc],b[ic]);
    }
    trsm(A(ic,ic),b[ic],'l');
  }
  stop("Forward substitution");
  start("Backward substitution");
  for (int ic=Nc-1; ic>=0; ic--) {
    for (int jc=Nc-1; jc>ic; jc--) {
      gemm(A(ic,jc),b[jc],b[ic]);
    }
    trsm(A(ic,ic),b[ic],'u');
  }
  stop("Backward substitution");
  double diff = 0, norm = 0;
  for (int ic=0; ic<Nc; ic++) {
    diff += (x.dense(ic) - b.dense(ic)).norm();
    norm += x.dense(ic).norm();
  }
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
