#include "mpi_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "dense.h"
#include "grid.h"
#include "print.h"
#include "timer.h"
#include <vector>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 64;
  int Nb = 4;
  int Nc = N / Nb;
  std::vector<int> ipiv(Nb);
  std::vector<double> randx(N);
  std::vector<Dense> x(Nc);
  std::vector<Dense> b(Nc);
  std::vector<Dense> A(Nc*Nc);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  print("Time");
  start("Init matrix");
  for (int ic=0; ic<Nc; ic++) {
    x[ic].resize(Nb);
    b[ic].resize(Nb);
    for (int ib=0; ib<Nb; ib++) {
      x[ic][ib] = randx[Nb*ic+ib];
      b[ic][ib] = 0;
    }
  }
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      A[Nc*ic+jc].resize(Nb,Nb);
      for (int ib=0; ib<Nb; ib++) {
        for (int jb=0; jb<Nb; jb++) {
          A[Nc*ic+jc](ib,jb) = 1 / (std::abs(x[ic][ib] - x[jc][jb]) + 1e-3);
          b[ic][ib] += A[Nc*ic+jc](ib,jb) * x[jc][jb];
        }
      }
    }
  }
  stop("Init matrix");
  start("LU decomposition");
  for (int ic=0; ic<Nc; ic++) {
    start("-DGETRF");
    A[Nc*ic+ic].getrf(ipiv);
    stop("-DGETRF", false);
    for (int jc=ic+1; jc<Nc; jc++) {
      start("-DTRSM");
      A[Nc*ic+jc].trsm(A[Nc*ic+ic],'l');
      A[Nc*jc+ic].trsm(A[Nc*ic+ic],'u');
      stop("-DTRSM", false);
    }
    for (int jc=ic+1; jc<Nc; jc++) {
      for (int kc=ic+1; kc<Nc; kc++) {
        start("-DGEMM");
        A[Nc*jc+kc].gemm(A[Nc*ic+kc], A[Nc*jc+ic]);
        stop("-DGEMM", false);
      }
    }
  }
  stop("LU decomposition");
  print2("-DGETRF");
  print2("-DTRSM");
  print2("-DGEMM");
  start("Forward substitution");
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<ic; jc++) {
      b[ic].gemv(A[Nc*ic+jc], b[jc]);
    }
    b[ic].trsm(A[Nc*ic+ic],'l');
  }
  stop("Forward substitution");
  start("Backward substitution");
  for (int ic=Nc-1; ic>=0; ic--) {
    for (int jc=Nc-1; jc>ic; jc--) {
      b[ic].gemv(A[Nc*ic+jc], b[jc]);
    }
    b[ic].trsm(A[Nc*ic+ic],'u');
  }
  stop("Backward substitution");

  double diff = 0, norm = 0;
  for (int ic=0; ic<Nc; ic++) {
    for (int ib=0; ib<Nb; ib++) {
      diff += (x[ic][ib] - b[ic][ib]) * (x[ic][ib] - b[ic][ib]);
      norm += x[ic][ib] * x[ic][ib];
    }
  }
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
