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

using namespace hicma;

int main(int argc, char** argv) {
  int N = 12;
  int Nb = 4;
  int Nc = N / Nb;
  std::vector<double> randx(N);
  double diff, norm;
  Dense Id(identity, randx, N, N);
  for(int i = 0; i < N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  Hierarchical A(Nc, Nc);
  Hierarchical T(Nc, Nc);
  for(int ic = 0; ic < Nc; ic++) {
    for(int jc = 0; jc < Nc; jc++) {
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      A(ic, jc) = Aij;
      //Fill T with zeros
      Dense Tij(Aij.dim[1], Aij.dim[1]);
      T(ic, jc) = Tij;
    }
  }
  Hierarchical ACpy(A);
  start("QR decomposition");
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
  stop("QR decomposition");

  //Take upper triangular part of A as R
  Dense DR(A);
  for(int i = 0; i < DR.dim[0]; i++) {
    for(int j = 0; j < i; j++) {
      DR(i, j) = 0.0;
    }
  }
  //Apply Q to Id to obtain Q
  Hierarchical Q(Id, Nc, Nc);
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
  // print("A after");
  // Dense(A).print();
  Dense DQ(Q);
  // print("R");
  // DR.print();
  Dense QR(N, N);
  QR.gemm(DQ, DR, 1, 0);
  diff = (Dense(ACpy) - QR).norm();
  norm = ACpy.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  Dense QtQ(N, N);
  QtQ.gemm(DQ, DQ, CblasTrans, CblasNoTrans, 1, 0);
  diff = (QtQ - Id).norm();
  norm = Id.norm();
  print("Orthogonality of Q");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}


