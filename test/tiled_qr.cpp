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
  int N = argc > 1 ? atoi(argv[1]) : 256;
  int Nb = argc > 2 ? atoi(argv[2]) : 32;
  int matCode = argc > 5 ? atoi(argv[5]) : 0;
  int Nc = N / Nb;
  std::vector<std::vector<double>> randpts;
  Dense DA;

  if(matCode == 0 || matCode == 1) { //Laplace1D or Helmholtz1D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  }
  else if(matCode == 2) { //Ill-conditioned Cauchy2D (Matrix A3 From HODLR Paper)
    randpts.push_back(equallySpacedVector(N, -4.25, 998.25));
    randpts.push_back(equallySpacedVector(N, -2.15, 999.45));
  }
  else { //Ill-conditioned generated from DLATMS
    //Configurations
    char dist = 'U'; //Uses uniform distribution when generating random SV
    std::vector<int> iseed{ 1, 23, 456, 789 };
    char sym = 'N'; //Generate symmetric matrix
    double dmax = 1.0;
    int kl = N-1;
    int ku = N-1;
    char pack = 'N';

    std::vector<double> d(N, 0.0); //Singular values to be used
    int mode = 1; //See docs
    double _cond = 1e+12; //Condition number of generated matrix
    Dense _DA(N, N);
    latms(N, N, dist, iseed, sym, d, mode, _cond, dmax, kl, ku, pack, _DA);
    DA = std::move(_DA);

    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  }

  Hierarchical A(Nc, Nc);
  Hierarchical T(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  for(int ic = 0; ic < Nc; ic++) {
    for(int jc = 0; jc < Nc; jc++) {
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
      else {
        Dense _Aij(zeros, randpts[0], Nb, Nb, Nb*ic, Nb*jc, ic, jc, 1);
        getSubmatrix(DA, Nb, Nb, Nb*ic, Nb*jc, _Aij);
        Aij = std::move(_Aij);
      }
      Dense Qij(identity, randpts[0], Nb, Nb, Nb*ic, Nb*jc);
      Dense Tij(zeros, randpts[0], Nb, Nb);
      A(ic, jc) = Aij;
      Q(ic, jc) = Qij;
      T(ic, jc) = Tij;
    }
  }
  //Compute condition number of A
  print("Cond(A)", cond(Dense(A)), false);

  // For residual measurement
  double diff, norm;
  Dense x(N); x = 1.0;
  Dense Ax(N);
  gemm(A, x, Ax, 1, 1);

  print("Time");
  start("QR decomposition");
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
  stop("QR decomposition");
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
  //Residual
  Dense Rx(N);
  gemm(A, x, Rx, 1, 1);
  Dense QRx(N);
  gemm(Q, Rx, QRx, 1, 1);
  diff = (Ax - QRx).norm();
  norm = Ax.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  //Orthogonality
  for(int i=0; i<Nc; i++)
    for(int j=0; j<Nc; j++) {
      if(i != j) {
        Dense Qij(Q(i, j));
        Dense S(std::min(Qij.dim[0], Qij.dim[1]));
        Qij.svd(S);
        std::cout <<"Q(" <<i <<"," <<j <<").S" <<std::endl;
        S.print();
      }
    }
  Dense Qx(N);
  gemm(Q, x, Qx, 1, 1);
  Dense QtQx(N);
  Q.transpose();
  gemm(Q, Qx, QtQx, 1, 1);
  diff = (QtQx - x).norm();
  norm = (double)N;
  print("Orthogonality");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}


