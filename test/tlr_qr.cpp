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
#include <cassert>

#include "yorel/multi_methods.hpp"

using namespace hicma;

void getSubmatrix(const Dense& A, int ni, int nj, int i_begin, int j_begin, Dense& out) {
  assert(out.dim[0] == ni);
  assert(out.dim[1] == nj);
  for(int i=0; i<ni; i++)
    for(int j=0; j<nj; j++) {
      out(i, j) = A(i+i_begin, j+j_begin);
    }
}

int main(int argc, char** argv) {
  yorel::multi_methods::initialize();
  int N = argc > 1 ? atoi(argv[1]) : 256;
  int Nb = argc > 2 ? atoi(argv[2]) : 32;
  int rank = argc > 3 ? atoi(argv[3]) : 16;
  double admis = argc > 4 ? atof(argv[4]) : 0;
  int lra = argc > 5 ? atoi(argv[5]) : 1; updateCounter("LRA", lra);
  int Nc = N / Nb;
  std::vector<std::vector<double>> randpts;
  randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  // randpts.push_back(equallySpacedVector(N, -4.25, 998.25));
  // randpts.push_back(equallySpacedVector(N, -2.15, 999.45));

  //Generate input matrix using LAPACK DLATMS
  // char dist = 'U'; //Uses uniform distribution when generating random SV
  // std::vector<int> iseed{ 1, 23, 456, 789 };
  // char sym = 'N'; //Generate symmetric matrix
  // std::vector<double> d(N, 0.0); //Singular values to be used
  // // d[0] = 1.0;
  // // d[1] = 1.0;
  // // d[2] = 1e-15;
  // int mode = 1; //See docs
  // double _cond = 1e+12; //Condition number of generated matrix
  // double dmax = 1.0;
  // int kl = N-1;
  // int ku = N-1;
  // char pack = 'N';
  // Dense DA(N, N);
  // latms(N, N, dist, iseed, sym, d, mode, _cond, dmax, kl, ku, pack, DA);

  Hierarchical A(Nc, Nc);
  Hierarchical D(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  Hierarchical T(Nc, Nc);
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      int bi = Nb + ((ic == (Nc-1)) ? N % Nb : 0);
      int bj = Nb + ((jc == (Nc-1)) ? N % Nb : 0);
      Dense Aij(laplacend, randpts, bi, bj, Nb*ic, Nb*jc, ic, jc, 1);
      // getSubmatrix(DA, bi, bj, Nb*ic, Nb*jc, Aij);
      Dense Qij(identity, randpts[0], bi, bj, Nb*ic, Nb*jc, ic, jc, 1);
      Dense Tij(zeros, randpts[0], bj, bj);
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

  //Compute condition number of A
  // print("Cond(A)", cond(Dense(D)), false);

  // For residual measurement
  Dense x(N); x = 1.0;
  Dense Ax(N);
  gemm(A, x, Ax, 1, 1);
  //Approximation error
  double diff, norm;
  diff = (Dense(A) - Dense(D)).norm();
  norm = D.norm();
  print("BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);

  print("Time");
  resetCounter("Recompression");
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
  int additionAfterR = globalCounter["LR-addition"];
  int recompAfterR = globalCounter["Recompression"];
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

  int additionAfterQ = globalCounter["LR-addition"] - additionAfterR;
  int recompAfterQ = globalCounter["Recompression"] - recompAfterR;
  // std::cout <<"BLR dimension = " <<Nc <<"x" <<Nc <<std::endl;
  // print("LR addition (after building R)", additionAfterR);
  // print("LR addition (after building Q)", additionAfterQ);
  // print("LR addition (total)", additionAfterR + additionAfterQ);
  // print("Recompression (after building R)", recompAfterR);
  // print("Recompression (after building Q)", recompAfterQ);
  // print("Recompression (total)", recompAfterR + recompAfterQ);

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
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}


