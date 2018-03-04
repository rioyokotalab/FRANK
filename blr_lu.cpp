#include "mpi_utils.h"
#include <algorithm>
#include "boost_any_wrapper.h"
#include <cmath>
#include <cstdlib>
#include "dense.h"
#include "hierarchical.h"
#include "print.h"
#include "timer.h"
#include <vector>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 64;
  int Nb = 16;
  int Nc = N / Nb;
  int rank = 8;
  std::vector<double> randx(N);
  Hierarchical x(Nc);
  Hierarchical b(Nc);
  Hierarchical A(Nc,Nc);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  print("Time");
  start("Init matrix");
  for (int ic=0; ic<Nc; ic++) {
    Dense xi(Nb);
    Dense bj(Nb);
    for (int ib=0; ib<Nb; ib++) {
      xi[ib] = randx[Nb*ic+ib];
      bj[ib] = 0;
    }
    x[ic] = xi;
    b[ic] = bj;
  }
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      Dense Aij(Nb,Nb);
      for (int ib=0; ib<Nb; ib++) {
        for (int jb=0; jb<Nb; jb++) {
          Aij(ib,jb) = 1 / (std::abs(x.D(ic)[ib] - x.D(jc)[jb]) + 1e-3);
          b.D(ic)[ib] += Aij(ib,jb) * x.D(jc)[jb];
        }
      }
      if (ic == jc) {
        A(ic,jc) = Aij;
      } else {
        A(ic,jc) = Aij;
      }
    }
  }
  stop("Init matrix");
  start("LU decomposition");
  for (int ic=0; ic<Nc; ic++) {
    start("-DGETRF");
    std::vector<int> ipiv = getrf(A(ic,ic));
    stop("-DGETRF", false);
    for (int jc=ic+1; jc<Nc; jc++) {
      start("-DTRSM");
      trsm(A(ic,ic),A(ic,jc),'l');
      trsm(A(ic,ic),A(jc,ic),'u');
      stop("-DTRSM", false);
    }
    for (int jc=ic+1; jc<Nc; jc++) {
      for (int kc=ic+1; kc<Nc; kc++) {
        start("-DGEMM");
        gemm(A(ic,kc),A(jc,ic),A(jc,kc));
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
      gemv(A(ic,jc),b[jc],b[ic]);
    }
    trsm(A(ic,ic),b[ic],'l');
  }
  stop("Forward substitution");
  start("Backward substitution");
  for (int ic=Nc-1; ic>=0; ic--) {
    for (int jc=Nc-1; jc>ic; jc--) {
      gemv(A(ic,jc),b[jc],b[ic]);
    }
    trsm(A(ic,ic),b[ic],'u');
  }
  stop("Backward substitution");

  double diff = 0, norm = 0;
  for (int ic=0; ic<Nc; ic++) {
    for (int ib=0; ib<Nb; ib++) {
      diff += (x.D(ic)[ib] - b.D(ic)[ib]) * (x.D(ic)[ib] - b.D(ic)[ib]);
      norm += x.D(ic)[ib] * x.D(ic)[ib];
    }
  }
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
