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
#include <random>

#include "yorel/multi_methods.hpp"

using namespace hicma;

std::vector<std::vector<double>> generatePoints(int N, int dim) {
  std::mt19937 generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  std::vector<std::vector<double>> pts(dim, std::vector<double>(N));
  //Generate points from random distribution
  for(int i=0; i<N; i++) {
    for(int j=0; j<dim; j++) {
      pts[j][i] = distribution(generator);
    }
  }
  //Sort points
  for(int j=0; j<dim; j++) {
    std::sort(pts[j].begin(), pts[j].end());
  }
  return pts;
}

int main(int argc, char** argv) {
  yorel::multi_methods::initialize();
  int N = argc > 1 ? atoi(argv[1]) : 256;
  int Nb = argc > 2 ? atoi(argv[2]) : 32;
  int rank = argc > 3 ? atoi(argv[3]) : 16;
  double admis = argc > 4 ? atof(argv[4]) : 0;
  int Nc = N / Nb;
  std::vector<std::vector<double>> randpts = generatePoints(N, 1);

  Hierarchical A(Nc, Nc);
  Hierarchical D(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  Hierarchical T(Nc, Nc);
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      int bi = Nb + ((ic == (Nc-1)) ? N % Nb : 0);
      int bj = Nb + ((jc == (Nc-1)) ? N % Nb : 0);
      Dense Aij(laplacend, randpts, bi, bj, Nb*ic, Nb*jc, ic, jc, 1);
      Dense Qij(identity, randpts[0], bi, bj, Nb*ic, Nb*jc);
      Dense Tij(zeros, randpts[0], bj, bj);
      D(ic,jc) = Aij;
      T(ic,jc) = Tij;
      if (std::abs(ic-jc) <= (int)admis) {
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
  printXML(A);

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
  Dense DQ(Q);
  Dense QtQ(DQ.dim[0], DQ.dim[1]);
  gemm(DQ, DQ, QtQ, CblasTrans, CblasNoTrans, 1, 0);
  Dense Id(identity, randpts[0], QtQ.dim[0], QtQ.dim[1]);
  diff = (QtQ - Id).norm();
  norm = Id.norm();
  print("Orthogonality");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}


