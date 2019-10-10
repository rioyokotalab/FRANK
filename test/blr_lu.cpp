#include "hicma/any.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cmath>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main(int argc, char** argv) {
  yorel::multi_methods::initialize();
  int N = 64;
  int Nb = 16;
  int Nc = N / Nb;
  int rank = 8;
  int admis = 1;
  std::vector<double> randx(N);
  Hierarchical x(Nc);
  Hierarchical b(Nc);
  Hierarchical A(Nc,Nc);
  Hierarchical D(Nc,Nc);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
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
      if (std::abs(ic - jc) <= admis) {
        A(ic,jc) = Aij;
      }
      else {
        rsvd_push(A(ic,jc), Aij, rank);
      }
    }
  }
  rsvd_batch();
  double diff = 0, norm = 0;
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      diff += (Dense(A(ic,jc)) - D(ic,jc)).norm();
      norm += D(ic,jc).norm();
    }
  }
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  print("Time");
  gemm(A, x, b, 1, 1);
  gemm_batch();
  stop("Init matrix");
  start("LU decomposition");
  for (int ic=0; ic<Nc; ic++) {
    getrf(A(ic,ic));
    for (int jc=ic+1; jc<Nc; jc++) {
      trsm(A(ic,ic), A(ic,jc),'l');
      trsm(A(ic,ic), A(jc,ic),'u');
    }
    for (int jc=ic+1; jc<Nc; jc++) {
      for (int kc=ic+1; kc<Nc; kc++) {
        gemm(A(jc,ic),A(ic,kc),A(jc,kc),-1,1);
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
      gemm(A(ic,jc),b[jc],b[ic],-1,1);
    }
    trsm(A(ic,ic), b[ic],'l');
  }
  stop("Forward substitution");
  printTime("-DTRSM");
  printTime("-DGEMM");
  start("Backward substitution");
  for (int ic=Nc-1; ic>=0; ic--) {
    for (int jc=Nc-1; jc>ic; jc--) {
      gemm(A(ic,jc),b[jc],b[ic],-1,1);
    }
    trsm(A(ic,ic), b[ic],'u');
  }
  stop("Backward substitution");
  printTime("-DTRSM");
  printTime("-DGEMM");
  diff = (Dense(x) - Dense(b)).norm();
  norm = x.norm();
  print("LU Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
