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
  Hierarchical T(Nc, Nc);
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
      Dense Qij(identity, randpts[0], Nb, Nb, Nb*ic, Nb*jc, ic, jc, 1);
      Dense Tij(zeros, randpts[0], Nb, Nb);
      D(ic,jc) = Aij;
      T(ic,jc) = Tij;
      if (std::abs(ic - jc) <= (int)admis) {
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

  // For residual measurement
  Dense x(N); x = 1.0;
  Dense Ax(N);
  gemm(A, x, Ax, 1, 0);
  //Approximation error
  double diff, l2;
  diff = norm(Dense(A) - Dense(D));
  l2 = norm(D);
  print("BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/l2), false);

  print("Time");
  resetCounter("LR-addition");
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

  printCounter("LR-addition");

  //Build R: Take upper triangular part of modified A
  for(int i=0; i<A.dim[0]; i++) {
    for(int j=0; j<=i; j++) {
      if(i == j) //Diagonal must be dense, zero lower-triangular part
        zero_lowtri(A(i, j));
      else
        zero_whole(A(i, j));
    }
  }
  //Residual
  Dense Rx(N);
  gemm(A, x, Rx, 1, 0);
  Dense QRx(N);
  gemm(Q, Rx, QRx, 1, 0);
  diff = norm(Ax - QRx);
  l2 = norm(Ax);
  print("Residual");
  print("Rel. Error (operator norm)", std::sqrt(diff/l2), false);
  //Orthogonality
  Dense Qx(N);
  gemm(Q, x, Qx, 1, 0);
  Dense QtQx(N);
  transpose(Q);
  gemm(Q, Qx, QtQx, 1, 0);
  diff = norm(QtQx - x);
  l2 = (double)N;
  print("Orthogonality");
  print("Rel. Error (operator norm)", std::sqrt(diff/l2), false);
  return 0;
}


