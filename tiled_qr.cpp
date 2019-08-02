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
  int N = 8;
  int Nb = 4;
  int Nc = N / Nb;
  int Nbs = Nb;
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
  Hierarchical _A(A);
  start("QR decomposition");
  for(int k = 0; k < Nc; k++) {
    Dense Akk(A(k, k));
    Dense Tkk(T(k, k));
    Akk.geqrt(Tkk);
    for(int j = k+1; j < Nc; j++) {
      Dense Akj(A(k, j));
      Akj.larfb(Akk, Tkk, true);
      A(k, j) = Akj;
    }
    T(k, k) = Tkk;
    for(int i = k+1; i < Nc; i++) {
      Dense Aik(A(i, k));
      Dense Tik(T(i, k));
      Aik.tpqrt(Akk, Tik);
      A(i, k) = Aik;
      T(i, k) = Tik;
      for(int j = k+1; j < Nc; j++) {
        Dense Aij(A(i, j));
        Dense Akj(A(k, j));
        Aij.tpmqrt(Akj, Aik, Tik, true);
        A(i, j) = Aij;
        A(k, j) = Akj;
      }
    }
    A(k, k) = Akk;
  }
  stop("QR decomposition");
  //Take R from upper triangular part of A
  Dense DA(A);
  Dense DR(zeros, randx, N, N);
  for(int i = 0; i < N; i++) {
    for(int j = i; j < N; j++) {
      DR(i, j) = DA(i, j);
    }
  }
  //Build Q
  Dense _Q(identity, randx, N, N);
  Hierarchical Q(_Q, Nc, Nc);
  for(int k = Nc-1; k >= 0; k--) {
    for(int i = Nc-1; i > k; i--) {
      Dense Yik(A(i, k));
      Dense Tik(T(i, k));
      for(int j = Nc-1; j >= k; j--) {
        Dense Qij(Q(i, j));
        Dense Qkj(Q(k, j));
        Qij.tpmqrt(Qkj, Yik, Tik, false);
        Q(i, j) = Qij;
        Q(k, j) = Qkj;
      }
    }
    Dense Qkk(Q(k, k));
    Dense Ykk(A(k, k));
    Dense Tkk(T(k, k));
    Qkk.larfb(Ykk, Tkk, false);
    Q(k, k) = Qkk;
    for(int j = k+1; j < Nc; j++) {
      Dense Qkj(Q(k, j));
      Qkj.larfb(Ykk, Tkk);
      Q(k, j) = Qkj;
    }
  }
  Dense DQ(Q);
  Dense QR(N, N);
  QR.gemm(DQ, DR, 1, 1);
  diff = (Dense(_A) - QR).norm();
  norm = _A.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  Dense QQt(N, N);
  QQt.gemm(Q, Q, CblasNoTrans, CblasTrans, 1, 1);
  diff = (QQt - Id).norm();
  norm = Id.norm();
  print("Orthogonality of Q");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}


