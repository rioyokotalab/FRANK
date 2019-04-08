#include "any.h"
#include "low_rank.h"
#include "hierarchical.h"
#include "functions.h"
#include "batch.h"
#include "print.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 256;
  int Nb = 16;
  int Nc = N / Nb;
  int rank = 8;
  std::vector<double> randx(N);
  Hierarchical A(Nc, Nc);
  Hierarchical D(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  Hierarchical R(Nc, Nc);
  for(int i = 0; i < N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      Dense Rij(zeros, randx, Nb, Nb, Nb*ic, Nb*jc);
      D(ic,jc) = Aij;
      if (std::abs(ic - jc) <= 1) {
        A(ic,jc) = Aij;
        R(ic,jc) = Rij;
      }
      else {
        rsvd_push(A(ic,jc), Aij, rank);
        rsvd_push(R(ic,jc), Rij, rank);
      }
    }
  }
  rsvd_batch();
  double diff = 0, norm = 0;
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      if(A(ic,jc).is(HICMA_LOWRANK)) {
        diff += (Dense(A(ic,jc)) - D(ic,jc)).norm();
        norm += D(ic,jc).norm();
      }
    }
  }
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  print("Time");
  Hierarchical _A(A); //Copy of A
  Hierarchical QR(R);
  start("QR decomposition");
  for(int j=0; j<Nc; j++) {
    Hierarchical Aj(Nc, 1);
    Hierarchical Qsj(Nc, 1);
    for(int i=0; i<Nc; i++) {
      Aj(i, 0) = A(i, j);
      Qsj(i, 0) = A(i, j);
    }
    Hierarchical Rjj(1, 1);
    Aj.blr_col_qr(Qsj, Rjj);
    R(j, j) = Rjj(0, 0);
    //Copy column of Qsj to Q
    for(int i = 0; i < Nc; i++) {
      Q(i, j) = Qsj(i, 0);
    }
    //Transpose of Qsj to be used in processing next columns
    Hierarchical TrQsj(Qsj);
    TrQsj.transpose();
    //Process next columns
    for(int k=j+1; k<Nc; k++) {
      //Take k-th column
      Hierarchical Ak(Nc, 1);
      for(int i=0; i<Nc; i++) {
        Ak(i, 0) = A(i, k);
      }
      Hierarchical Rjk(1, 1);
      Rjk(0, 0) = R(j, k);
      Rjk.gemm(TrQsj, Ak, 1, 1); //Rjk = Q*j^T x A*k
      R(j, k) = Rjk(0, 0);
      Ak.gemm(Qsj, Rjk, -1, 1); //A*k = A*k - Q*j x Rjk
      for(int i=0; i<Nc; i++) {
        A(i, k) = Ak(i, 0);
      }
    }
  }
  stop("QR decomposition");
  printTime("-DGEQRF");
  printTime("-DGEMM");
  QR.gemm(Q, R, 1, 1);
  diff = (Dense(_A) - Dense(QR)).norm();
  norm = _A.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  //Orthogonality checking (max norm)
  double maxNorm = 0.0;
  for(int i=0; i<Nc; i++) {
    for(int j=0; j<Nc; j++) {
      Hierarchical Qi(Nc, 1);
      Hierarchical Qj(Nc, 1);
      for(int k=0; k<Nc; k++) {
        Qi(k, 0) = Q(k, i);
        Qj(k, 0) = Q(k, j);
      }
      Qi.transpose();
      Hierarchical QiTQj(1, 1);
      QiTQj(0, 0) = Dense(Nb, Nb);
      QiTQj.gemm(Qi, Qj, 1, 1);
      if(i == j) {
        Dense Id(identity, randx, Nb, Nb);
        maxNorm = fmax(maxNorm, (Dense(QiTQj) - Id).norm());
      }
      else {
        maxNorm = fmax(maxNorm, QiTQj.norm());
      }
    }
  }
  print("Orthogonality (maximum off-diagonal blocks norm)", maxNorm, false);
  return 0;
}


