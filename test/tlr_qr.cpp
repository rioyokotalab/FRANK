#include "hicma/hicma.h"

#include "yorel/yomm2/cute.hpp"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>


using namespace hicma;

int main(int argc, char** argv) {
  yorel::yomm2::update_methods();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t Nb = argc > 2 ? atoi(argv[2]) : 32;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 16;
  double admis = argc > 4 ? atof(argv[4]) : 0;
  int64_t matCode = argc > 5 ? atoi(argv[5]) : 0;
  int64_t lra = argc > 6 ? atoi(argv[6]) : 2; updateCounter("LRA", lra);
  int64_t Nc = N / Nb;
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
  for (int64_t ic=0; ic<Nc; ic++) {
    for (int64_t jc=0; jc<Nc; jc++) {
      Dense Aij;
      if(matCode == 0) {
        Dense _Aij(laplacend, randpts, Nb, Nb, Nb*ic, Nb*jc);
        Aij = std::move(_Aij);
      }
      else if(matCode == 1) {
        Dense _Aij(helmholtznd, randpts, Nb, Nb, Nb*ic, Nb*jc);
        Aij = std::move(_Aij);
      }
      else if(matCode == 2) {
        Dense _Aij(cauchy2d, randpts, Nb, Nb, Nb*ic, Nb*jc);
        Aij = std::move(_Aij);
      }
      Dense Qij(identity, randpts[0], Nb, Nb, Nb*ic, Nb*jc);
      Dense Tij(zeros, randpts[0], Nb, Nb);
      D(ic,jc) = Aij;
      T(ic,jc) = Tij;
      if (std::abs(ic - jc) <= (int64_t)admis) {
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
  Dense Ax = gemm(A, x);

  print("BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, D), false);

  print("Time");
  resetCounter("LR-addition");
  timing::start("BLR QR decomposition");
  for(int64_t k = 0; k < Nc; k++) {
    geqrt(A(k, k), T(k, k));
    for(int64_t j = k+1; j < Nc; j++) {
      larfb(A(k, k), T(k, k), A(k, j), true);
    }
    for(int64_t i = k+1; i < Nc; i++) {
      tpqrt(A(k, k), A(i, k), T(i, k));
      for(int64_t j = k+1; j < Nc; j++) {
        tpmqrt(A(i, k), T(i, k), A(k, j), A(i, j), true);
      }
    }
  }
  timing::stopAndPrint("BLR QR decomposition");
  //Build Q: Apply Q to Id
  for(int64_t k = Nc-1; k >= 0; k--) {
    for(int64_t i = Nc-1; i > k; i--) {
      for(int64_t j = k; j < Nc; j++) {
        tpmqrt(A(i, k), T(i, k), Q(k, j), Q(i, j), false);
      }
    }
    for(int64_t j = k; j < Nc; j++) {
      larfb(A(k, k), T(k, k), Q(k, j), false);
    }
  }

  printCounter("LR-addition");

  //Build R: Take upper triangular part of modified A
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<=i; j++) {
      if(i == j) //Diagonal must be dense, zero lower-triangular part
        zero_lowtri(A(i, j));
      else
        zero_whole(A(i, j));
    }
  }
  //Residual
  Dense Rx = gemm(A, x);
  Dense QRx = gemm(Q, Rx);
  print("Residual");
  print("Rel. Error (operator norm)", l2_error(QRx, Ax), false);
  //Orthogonality
  Dense Qx = gemm(Q, x);
  transpose(Q);
  Dense QtQx = gemm(Q, Qx);
  print("Orthogonality");
  print("Rel. Error (operator norm)", l2_error(QtQx, x), false);
  return 0;
}
