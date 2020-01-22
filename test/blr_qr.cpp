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
  std::mt19937 generator(time(0));
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
  Hierarchical R(Nc, Nc);
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      int bi = Nb + ((ic == (Nc-1)) ? N % Nb : 0);
      int bj = Nb + ((jc == (Nc-1)) ? N % Nb : 0);
      Dense Aij(laplacend, randpts, bi, bj, Nb*ic, Nb*jc, ic, jc, 1);
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
      gemm(TrQsj, Ak, R(j, k), 1, 1); //Rjk = Q*j^T x A*k
      gemm(Qsj, R(j, k), Ak, -1, 1); //A*k = A*k - Q*j x Rjk
      for(int i=0; i<Nc; i++) {
        A(i, k) = Ak(i, 0);
      }
    }
  }
  stop("BLR QR decomposition");

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
  Dense DQ(Q);
  Dense QtQ(DQ.dim[1], DQ.dim[1]);
  gemm(DQ, DQ, QtQ, CblasTrans, CblasNoTrans, 1, 0);
  Dense Id(identity, randpts[0], QtQ.dim[0], QtQ.dim[1]);
  diff = (QtQ - Id).norm();
  norm = Id.norm();
  print("Orthogonality");
  print("Rel. L2 Orthogonality", std::sqrt(diff/norm), false);
  return 0;
}


