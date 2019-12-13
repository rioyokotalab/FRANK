#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"
#include "hicma/util/counter.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main(int argc, char** argv) {
  yorel::multi_methods::initialize();
  int N = argc > 1 ? atoi(argv[1]) : 256;
  int Nb = argc > 2 ? atoi(argv[2]) : 32;
  int rank = argc > 3 ? atoi(argv[3]) : 16;
  double admis = argc > 4 ? atof(argv[4]) : 0;
  int matCode = argc > 5 ? atoi(argv[5]) : 0;
  int lra = argc > 6 ? atoi(argv[6]) : 2; updateCounter("LRA", lra);
  int Nc = N / Nb;
  std::vector<std::vector<double>> randpts;
  updateCounter("LR_ADDITION_COUNTER", 1); //Enable LR addition counter

  if(matCode == 0 || matCode == 1) { //Laplace1D or Helmholtz1D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  }
  else { //Ill-conditioned Cauchy2D (Matrix A3 From HODLR Paper)
    randpts.push_back(equallySpacedVector(N, -1.25, 998.25));
    randpts.push_back(equallySpacedVector(N, -0.15, 999.45));
  }

  Dense DA;
  Hierarchical A(Nc, Nc);
  Hierarchical D(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  Hierarchical R(Nc, Nc);
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      Dense Aij;
      if(matCode == 0) {
        Dense _Aij(laplacend, randpts, Nb, Nb, Nb*ic, Nb*jc, ic, jc, 1);
        Aij = std::move(_Aij);
      }
      else if(matCode == 1) {
        Dense _Aij(helmholtznd, randpts, Nb, Nb, Nb*ic, Nb*jc, ic, jc, 1);
        Aij = std::move(_Aij);
      }
      else if(matCode == 2) {
        Dense _Aij(cauchy2d, randpts, Nb, Nb, Nb*ic, Nb*jc, ic, jc, 1);
        Aij = std::move(_Aij);
      }
      Dense Rij(zeros, randpts[0], Nb, Nb, Nb*ic, Nb*jc, ic, jc, 1);
      D(ic,jc) = Aij;
      if (std::abs(ic - jc) <= (int)admis) {
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

  //For residual measurement
  Dense x(N);
  x = 1.0;
  Dense Ax(N);
  gemm(A, x, Ax, 1, 0);
  //Approximation error
  double diff, norm;
  diff = (Dense(A) - Dense(D)).norm();
  norm = D.norm();
  print("Ida BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);

  print("Time");
  start("BLR QR decomposition");
  resetCounter("LR-addition");
  for(int j=0; j<Nc; j++) {
    Hierarchical Aj(Nc, 1);
    Hierarchical Qsj(Nc, 1);
    for(int i=0; i<Nc; i++) {
      Aj(i, 0) = A(i, j);
      Qsj(i, 0) = A(i, j);
    }
    Hierarchical Rjj(1, 1);
    Aj.blr_col_qr(Qsj, Rjj);
    R(j, j) = std::move(Rjj(0, 0));
    //Copy column of Qsj to Q
    for(int i = 0; i < Nc; i++) {
      Q(i, j) = Qsj(i, 0);
    }
    //Transpose of Qsj to be used in computing Rjk
    Hierarchical TrQsj(Qsj);
    transpose(TrQsj);
    //Process next columns
    for(int k=j+1; k<Nc; k++) {
      //Take k-th column
      Hierarchical Ak(Nc, 1);
      for(int i=0; i<Nc; i++) {
        Ak(i, 0) = A(i, k);
      }
      gemm(TrQsj, Ak, R(j, k), 1, 0); //Rjk = Q*j^T x A*k
      gemm(Qsj, R(j, k), Ak, -1, 1); //A*k = A*k - Q*j x Rjk
      for(int i=0; i<Nc; i++) {
        A(i, k) = std::move(Ak(i, 0));
      }
    }
  }
  stop("BLR QR decomposition");
  printCounter("LR-addition");

  //Residual
  Dense Rx(N);
  gemm(R, x, Rx, 1, 0);
  Dense QRx(N);
  gemm(Q, Rx, QRx, 1, 0);
  diff = (Ax - QRx).norm();
  norm = Ax.norm();
  print("Residual");
  print("Rel. Error (operator norm)", std::sqrt(diff/norm), false);
  //Orthogonality
  Dense Qx(N);
  gemm(Q, x, Qx, 1, 0);
  Dense QtQx(N);
  Q.transpose();
  gemm(Q, Qx, QtQx, 1, 0);
  diff = (QtQx - x).norm();
  norm = (double)N;
  print("Orthogonality");
  print("Rel. Error (operator norm)", std::sqrt(diff/norm), false);
  return 0;
}

