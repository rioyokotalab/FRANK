#include "any.h"
#include "low_rank.h"
#include "hierarchical.h"
#include "functions.h"
#include "batch.h"
#include "print.h"
#include "timer.h"

#include <algorithm>
#include <cmath>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 64;
  int Nb = 16;
  int Nc = N / Nb;
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
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      A(ic,jc) = Aij;
    }
  }
  b.gemm(A, x, 1, 1);
  gemm_batch();
  stop("Init matrix");
  start("LU decomposition");
  for (int ic=0; ic<Nc; ic++) {
    A(ic,ic).getrf();
    for (int jc=ic+1; jc<Nc; jc++) {
      A(ic,jc).trsm(A(ic,ic),'l');
      A(jc,ic).trsm(A(ic,ic),'u');
    }
    for (int jc=ic+1; jc<Nc; jc++) {
      for (int kc=ic+1; kc<Nc; kc++) {
        A(jc,kc).gemm(A(jc,ic),A(ic,kc));
      }
    }
  }
  stop("LU decomposition");
  printTime("-DGETRF");
  printTime("-DTRSM");
  printTime("-DGEMM");
  start("Forward substitution");
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<ic; jc++) {
      b[ic].gemm(A(ic,jc),b[jc]);
    }
    b[ic].trsm(A(ic,ic),'l');
  }
  stop("Forward substitution");
  printTime("-DTRSM");
  printTime("-DGEMM");
  start("Backward substitution");
  for (int ic=Nc-1; ic>=0; ic--) {
    for (int jc=Nc-1; jc>ic; jc--) {
      b[ic].gemm(A(ic,jc),b[jc]);
    }
    b[ic].trsm(A(ic,ic),'u');
  }
  stop("Backward substitution");
  printTime("-DTRSM");
  printTime("-DGEMM");
  double diff = (Dense(x) - Dense(b)).norm();
  double norm = x.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
