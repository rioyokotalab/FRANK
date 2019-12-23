#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main(int argc, char** argv) {
  yorel::multi_methods::initialize();
  int N = 256;
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
  Hierarchical T(Nc, Nc);
  Hierarchical QR(Nc, Nc);
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      Dense Qij(identity, randx, Nb, Nb, Nb*ic, Nb*jc);
      Dense Tij(zeros, randx, Nb, Nb, Nb*ic, Nb*jc);
      D(ic,jc) = Aij;
      T(ic,jc) = Tij;
      if (std::abs(ic - jc) <= 0) {
        A(ic,jc) = Aij;
        Q(ic,jc) = Qij;
        QR(ic,jc) = Tij;
      }
      else {
        rsvd_push(A(ic,jc), Aij, rank);
        rsvd_push(Q(ic,jc), Qij, rank);
        rsvd_push(QR(ic,jc), Tij, rank);
      }
    }
  }
  rsvd_batch();
  Hierarchical A_copy(A);
  double diff, norm;
  diff = (Dense(A) - Dense(D)).norm();
  norm = D.norm();
  print("BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);

  print("Time");
  start("BLR QR decomposition");
  for(int k = 0; k < Nc; k++) {
    geqrt(A(k, k), T(k, k));
    for(int j = k+1; j < Nc; j++) {
      larfb(A(k, k), T(k, k), A(k, j), true);
    }
    for(int i = k+1; i < Nc; i++) {
      tpqrt(A(k, k), A(i, k), T(i, k));
      for(int j = k+1; j < Nc; j++) {
        tpmqrt(A(i, k), T(i, k), A(k, j), A(i, j), true);
      }
    }
  }
  stop("BLR QR decomposition");
  //Build Q: Apply Q to Id
  for(int k = Nc-1; k >= 0; k--) {
    for(int i = Nc-1; i > k; i--) {
      for(int j = k; j < Nc; j++) {
        tpmqrt(A(i, k), T(i, k), Q(k, j), Q(i, j), false);
      }
    }
    for(int j = k; j < Nc; j++) {
      larfb(A(k, k), T(k, k), Q(k, j), false);
    }
  }
  //Build R: Take upper triangular part of modified A
  for(int i=0; i<A.dim[0]; i++) {
    for(int j=0; j<=i; j++) {
      if(i == j) //Must be dense, zero lower-triangular part
        zero_lowtri(A(i, j));
      else
        zero_whole(A(i, j));
    }
  }

  gemm(Q, A, QR, 1, 1);
  diff = (Dense(A_copy) - Dense(QR)).norm();
  norm = A_copy.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  Dense DQ(Q);
  Dense QtQ(DQ.dim[0], DQ.dim[1]);
  gemm(DQ, DQ, QtQ, CblasTrans, CblasNoTrans, 1, 0);
  Dense Id(identity, randx, QtQ.dim[0], QtQ.dim[1]);
  diff = (QtQ - Id).norm();
  norm = Id.norm();
  print("Orthogonality");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}


