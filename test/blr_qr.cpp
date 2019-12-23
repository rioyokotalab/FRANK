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
      if (std::abs(ic - jc) <= 0) {
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
  Hierarchical A_copy(A);
  double diff, norm;
  diff = (Dense(A) - Dense(D)).norm();
  norm = D.norm();
  print("Ida BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);

  print("Time");
  Hierarchical QR(R);
  start("BLR QR decomposition");
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
      // Rjk(0, 0) = R(j, k);
      // gemm(TrQsj, Ak, Rjk, 1, 1); //Rjk = Q*j^T x A*k
      // R(j, k) = Rjk(0, 0);
      // gemm(Qsj, Rjk, Ak, -1, 1); //A*k = A*k - Q*j x Rjk
      gemm(TrQsj, Ak, R(j, k), 1, 1); //Rjk = Q*j^T x A*k
      gemm(Qsj, R(j, k), Ak, -1, 1); //A*k = A*k - Q*j x Rjk
      for(int i=0; i<Nc; i++) {
        A(i, k) = Ak(i, 0);
      }
    }
  }
  stop("BLR QR decomposition");
  gemm(Q, R, QR, 1, 1);
  diff = (Dense(A_copy) - Dense(QR)).norm();
  norm = A_copy.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  Dense DQ(Q);
  Dense QtQ(DQ.dim[1], DQ.dim[1]);
  gemm(DQ, DQ, QtQ, CblasTrans, CblasNoTrans, 1, 0);
  Dense Id(identity, randx, QtQ.dim[0], QtQ.dim[1]);
  diff = (QtQ - Id).norm();
  norm = Id.norm();
  print("Orthogonality");
  print("Rel. L2 Orthogonality", std::sqrt(diff/norm), false);
  return 0;
}


