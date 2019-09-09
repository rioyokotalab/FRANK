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
#include <lapacke.h>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 128;
  int Nb = 32;
  int Nc = N / Nb;
  int rank = 16;
  std::vector<double> randx(N);
  for(int i = 0; i < N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  Hierarchical A(Nc, Nc);
  Hierarchical D(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      Dense Qij(identity, randx, Nb, Nb, Nb*ic, Nb*jc);
      D(ic,jc) = Aij;
      if (std::abs(ic - jc) <= 0) {
        A(ic,jc) = Aij;
        Q(ic,jc) = Qij;
      }
      else {
        rsvd_push(A(ic,jc), Aij, rank);
        rsvd_push(Q(ic,jc), Qij, rank);
      }
    }
  }
  rsvd_batch();
  double diff, norm;
  diff = (Dense(A) - D).norm();
  norm = D.norm();
  print("BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);

  Dense Id(identity, randx, N, N);
  Dense Z(zeros, randx, N, N);
  Hierarchical T(Z, Nc, Nc);
  print("Time");
  start("BLR QR decomposition");
  for(int k = 0; k < Nc; k++) {
    A(k, k).geqrt(T(k, k));
    for(int j = k+1; j < Nc; j++) {
      A(k, j).larfb(A(k, k), T(k, k), true);
    }
    for(int i = k+1; i < Nc; i++) {
      A(i, k).tpqrt(A(k, k), T(i, k));
      for(int j = k+1; j < Nc; j++) {
        A(i, j).tpmqrt(A(k, j), A(i, k), T(i, k), true);
      }
    }
  }
  stop("BLR QR decomposition");
  //Build R: Take upper triangular part of A
  Dense DR(A);
  for(int i = 0; i < DR.dim[0]; i++) {
    for(int j = 0; j < std::min(i, DR.dim[1]); j++) {
      DR(i, j) = 0.0;
    }
  }
  //Build Q: Apply Q to Id
  for(int k = Nc-1; k >= 0; k--) {
    for(int i = Nc-1; i > k; i--) {
      for(int j = k; j < Nc; j++) {
        Q(i, j).tpmqrt(Q(k, j), A(i, k), T(i, k), false);
      }
    }
    for(int j = k; j < Nc; j++) {
      Q(k, j).larfb(A(k, k), T(k, k), false);
    }
  }
  Dense DQ(Q);
  Dense QR(N, N);
  QR.gemm(DQ, DR, 1, 0);
  diff = (Dense(D) - QR).norm();
  norm = D.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  Dense QtQ(N, N);
  QtQ.gemm(DQ, DQ, CblasTrans, CblasNoTrans, 1, 0);
  diff = (QtQ - Id).norm();
  norm = Id.norm();
  print("Orthogonality");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}


