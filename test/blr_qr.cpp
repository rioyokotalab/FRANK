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
  int lra = argc > 6 ? atoi(argv[6]) : 1; updateCounter("LRA", lra);
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
  Hierarchical D(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  Hierarchical R(Nc, Nc);
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      int bi = Nb + ((ic == (Nc-1)) ? N % Nb : 0);
      int bj = Nb + ((jc == (Nc-1)) ? N % Nb : 0);
      Dense Aij;
      if(matCode == 0) {
        Dense _Aij(laplacend, randpts, bi, bj, Nb*ic, Nb*jc, ic, jc, 1);
        Aij = std::move(_Aij);
      }
      else if(matCode == 1) {
        Dense _Aij(helmholtznd, randpts, bi, bj, Nb*ic, Nb*jc, ic, jc, 1);
        Aij = std::move(_Aij);
      }
      else if(matCode == 2) {
        Dense _Aij(cauchy2d, randpts, bi, bj, Nb*ic, Nb*jc, ic, jc, 1);
        Aij = std::move(_Aij);
      }
      else {
        Dense _Aij(zeros, randpts[0], bi, bj, Nb*ic, Nb*jc, ic, jc, 1);
        getSubmatrix(DA, bi, bj, Nb*ic, Nb*jc, _Aij);
        Aij = std::move(_Aij);
      }
      Dense Rij(zeros, randpts[0], bi, bj, Nb*ic, Nb*jc, ic, jc, 1);
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

  //Compute condition number of A
  // print("Cond(A)", cond(Dense(D)), false);

  //For residual measurement
  Dense x(N);
  x = 1.0;
  Dense Ax(N);
  gemm(A, x, Ax, 1, 1);
  //Approximation error
  double diff, norm;
  diff = (Dense(A) - Dense(D)).norm();
  norm = D.norm();
  print("Ida BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);

  print("Time");
  start("BLR QR decomposition");
  resetCounter("Recompression");
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
    Hierarchical TrQsj(Qsj); TrQsj.transpose();
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

  // std::cout <<"BLR dimension = " <<Nc <<"x" <<Nc <<std::endl;
  // printCounter("Recompression");
  // printCounter("LR-addition");

  //Residual
  Dense Rx(N);
  gemm(R, x, Rx, 1, 1);
  Dense QRx(N);
  gemm(Q, Rx, QRx, 1, 1);
  diff = (Ax - QRx).norm();
  norm = Ax.norm();
  print("Residual");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  //Orthogonality
  Dense Qx(N);
  gemm(Q, x, Qx, 1, 1);
  Dense QtQx(N);
  Q.transpose();
  gemm(Q, Qx, QtQx, 1, 1);
  diff = (QtQx - x).norm();
  norm = (double)N;
  print("Orthogonality");
  print("Rel. L2 Orthogonality", std::sqrt(diff/norm), false);
  return 0;
}


