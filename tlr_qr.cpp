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
  int N = 24;
  int Nb = 8;
  int Nc = N / Nb;
  int rank = 6;
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
  Hierarchical B(A);
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
  //Apply Q^T to A to obtain R
  Hierarchical R(B);
  for(int k = 0; k < Nc; k++) {
    for(int j = k; j < Nc; j++) {
      R(k, j).larfb(A(k, k), T(k, k), true);
    }
    for(int i = k+1; i < Nc; i++) {
      for(int j = k; j < Nc; j++) {
        R(i, j).tpmqrt(R(k, j), A(i, k), T(i, k), true);
      }
    }
  }
  //Apply Q to Id to obtain Q
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
  Dense DR(R);
  Dense DQ(Q);
  Dense QR(N, N);
  QR.gemm(DQ, DR, 1, 0);
  diff = (Dense(B) - QR).norm();
  norm = B.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  Dense QtQ(N, N);
  QtQ.gemm(DQ, DQ, CblasTrans, CblasNoTrans, 1, 0);
  diff = (QtQ - Id).norm();
  norm = Id.norm();
  print("Orthogonality");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);

  std::cout <<std::endl;

  //=========================Dense QR===========================
  print("Dense QR Decomposition");
  print("Time");
  Dense AD(laplace1d, randx, N, N);
  Dense _AD(AD);
  Dense TD(N, N);
  start("Dense QR decomposition");
  AD.geqrt(TD);
  stop("Dense QR decomposition");
  //Apply Q^T to A to obtain R
  Dense RD(_AD);
  RD.larfb(AD, TD, true);
  //Apply Q to Id to obtain Q
  Dense QD(identity, randx, N, N);
  QD.larfb(AD, TD, false);
  print("Accuracy");
  Dense QRD(N, N);
  QRD.gemm(QD, RD, 1, 0);
  diff = (_AD - QRD).norm();
  norm = _AD.norm();
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  Dense QtQD(N, N);
  QtQD.gemm(QD, QD, CblasTrans, CblasNoTrans, 1, 0);
  diff = (QtQD - Id).norm();
  norm = Id.norm();
  print("Orthogonality");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  //============================================================

  return 0;
}


