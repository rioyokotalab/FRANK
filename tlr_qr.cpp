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

void test_geqrt() {
  std::cout <<"Using LAPACK GEQRT" <<std::endl;
  double a[4*4] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  double t[4*4] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int m = 4;
  int n = 4;
  int lda = 4;
  int ldt = 4;
  LAPACKE_dgeqrt(LAPACK_ROW_MAJOR, m, n, n, a, lda, t, ldt);
  std::cout <<"A" <<std::endl;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      std::cout <<a[i*n+j] <<" ";
    }
    std::cout <<std::endl;
  }
  std::cout <<"T" <<std::endl;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      std::cout <<t[i*n+j] <<" ";
    }
    std::cout <<std::endl;
  }
}

void test_geqrt2() {
  std::cout <<"Using LAPACK GEQRT2" <<std::endl;
  double a[4*4] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  double t[4*4] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int m = 4;
  int n = 4;
  int lda = 4;
  int ldt = 4;
  LAPACKE_dgeqrt2(LAPACK_ROW_MAJOR, m, n, a, lda, t, ldt);
  std::cout <<"A" <<std::endl;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      std::cout <<a[i*n+j] <<" ";
    }
    std::cout <<std::endl;
  }
  std::cout <<"T" <<std::endl;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      std::cout <<t[i*n+j] <<" ";
    }
    std::cout <<std::endl;
  }
}

void test_geqrt3() {
  std::cout <<"Using LAPACK GEQRT3" <<std::endl;
  double a[4*4] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  double t[4*4] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int m = 4;
  int n = 4;
  int lda = 4;
  int ldt = 4;
  LAPACKE_dgeqrt3(LAPACK_ROW_MAJOR, m, n, a, lda, t, ldt);
  std::cout <<"A" <<std::endl;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      std::cout <<a[i*n+j] <<" ";
    }
    std::cout <<std::endl;
  }
  std::cout <<"T" <<std::endl;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      std::cout <<t[i*n+j] <<" ";
    }
    std::cout <<std::endl;
  }
}

void test_geqrf_larft() {
  std::cout <<"Using LAPACK GEQRF+LARFT" <<std::endl;
  double a[4*4] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  double t[4*4] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  double tau[4] = {0, 0, 0, 0};
  int m = 4;
  int n = 4;
  int lda = 4;
  int ldt = 4;
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
  LAPACKE_dlarft(LAPACK_ROW_MAJOR, 'F', 'C', n, n, a, lda, tau, t, ldt);
  std::cout <<"A" <<std::endl;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      std::cout <<a[i*n+j] <<" ";
    }
    std::cout <<std::endl;
  }
  std::cout <<"tau" <<std::endl;
  for(int i = 0; i < n; i++) {
    std::cout <<tau[i] <<" ";
  }
  std::cout <<std::endl;
  std::cout <<"T" <<std::endl;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      std::cout <<t[i*n+j] <<" ";
    }
    std::cout <<std::endl;
  }
}

int main(int argc, char** argv) {
  int N = 128;
  int Nb = 16;
  int Nc = N / Nb;
  int rank = 8;
  std::vector<double> randx(N);
  for(int i = 0; i < N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  Hierarchical A(Nc, Nc);
  Hierarchical D(Nc, Nc);
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      D(ic,jc) = Aij;
      if (std::abs(ic - jc) <= 1) {
        A(ic,jc) = Aij;
      }
      else {
        rsvd_push(A(ic,jc), Aij, rank);
      }
    }
  }
  rsvd_batch();
  double diff, norm;
  diff = (Dense(A) - D).norm();
  norm = D.norm();
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);

  Dense Id(identity, randx, N, N);
  Dense Z(zeros, randx, N, N);
  Hierarchical B(A);
  Hierarchical T(Z, Nc, Nc);
  print("Time");
  start("QR decomposition");
  for(int k = 0; k < Nc; k++) {
    // std::cout <<"A(" <<k+1 <<"," <<k+1 <<").geqrt(T(" <<k+1 <<"," <<k+1 <<"))" <<std::endl;
    A(k, k).geqrt(T(k, k));
    // std::cout <<"T(" <<k+1 <<"," <<k+1 <<")" <<std::endl;
    // T(k, k).print();
    for(int j = k+1; j < Nc; j++) {
      // std::cout <<"A(" <<k+1 <<"," <<j+1 <<").larfb(A(" <<k+1 <<"," <<k+1 <<"), T(" <<k+1 <<"," <<k+1 <<"), true)" <<std::endl;
      A(k, j).larfb(A(k, k), T(k, k), true);
    }
    for(int i = k+1; i < Nc; i++) {
      // std::cout <<"A(" <<i+1 <<"," <<k+1 <<").tpqrt(A(" <<k+1 <<"," <<k+1 <<"), T(" <<i+1 <<"," <<k+1 <<"))" <<std::endl;
      A(i, k).tpqrt(A(k, k), T(i, k));
      // std::cout <<"T(" <<i+1 <<"," <<k+1 <<")" <<std::endl;
      // T(i, k).print();
      for(int j = k+1; j < Nc; j++) {
        // std::cout <<"A(" <<i+1 <<"," <<j+1 <<").tpmqrt(A(" <<k+1 <<"," <<j+1 <<"), A(" <<i+1 <<"," <<k+1 <<"), T(" <<i+1 <<"," <<k+1 <<"), true)" <<std::endl;
        A(i, j).tpmqrt(A(k, j), A(i, k), T(i, k), true);
      }
    }
  }
  stop("QR decomposition");

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
  Dense DR(R);
  Dense DQ(Q);
  Dense QR(N, N);
  QR.gemm(DQ, DR, 1, 0);
  diff = (Dense(B) - QR).norm();
  norm = B.norm();
  // print("A");
  // Dense(D).print();
  // print("R");
  // DR.print();
  // print("QR");
  // QR.print();
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


