#include "any.h"
#include "low_rank.h"
#include "hierarchical.h"
#include "functions.h"
#include "batch_rsvd.h"
#include "print.h"
#include "timer.h"

#include <algorithm>
#include <cmath>

using namespace hicma;

int main(int argc, char** argv) {
  int N = atoi(argv[1]);
  int Nb = atoi(argv[2]);
  int Nc = N / Nb;
  int rank = 8;
  std::vector<double> randx(N);
  Hierarchical x(Nc);
  Hierarchical b(Nc);
  Hierarchical A(Nc,Nc);
  Hierarchical D(Nc,Nc);
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
      D(ic,jc) = Aij;
      if (std::abs(ic - jc) <= 1) {
        A(ic,jc) = Aij;
      }
      else {
        low_rank_push(A(ic,jc), Aij, rank);
      }
    }
  }
  batch_rsvd();
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      if(A(ic,jc).is(HICMA_LOWRANK)) static_cast<LowRank&>(*A(ic,jc).ptr).print();
    }
  }
  /*
  b.gemm(A,x);
  stop("Init matrix");
  start("LU decomposition");
  for (int ic=0; ic<Nc; ic++) {
    start("-DGETRF");
    A(ic,ic).getrf();
    stop("-DGETRF", false);
    for (int jc=ic+1; jc<Nc; jc++) {
      start("-DTRSM");
      A(ic,jc).trsm(A(ic,ic),'l');
      A(jc,ic).trsm(A(ic,ic),'u');
      stop("-DTRSM", false);
    }
    for (int jc=ic+1; jc<Nc; jc++) {
      for (int kc=ic+1; kc<Nc; kc++) {
        start("-DGEMM");
        A(jc,kc).gemm(A(jc,ic),A(ic,kc));
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
      b[ic].gemm(A(ic,jc),b[jc]);
    }
    b[ic].trsm(A(ic,ic),'l');
  }
  stop("Forward substitution");
  start("Backward substitution");
  for (int ic=Nc-1; ic>=0; ic--) {
    for (int jc=Nc-1; jc>ic; jc--) {
      b[ic].gemm(A(ic,jc),b[jc]);
    }
    b[ic].trsm(A(ic,ic),'u');
  }
  stop("Backward substitution");
  double diff = (Dense(x) + Dense(b)).norm();
  double norm = x.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  */
  return 0;
}
