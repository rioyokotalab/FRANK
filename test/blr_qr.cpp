#include "hicma/hicma.h"

#include "yorel/yomm2/cute.hpp"

#include <algorithm>
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
  Hierarchical R(Nc, Nc);
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
      Dense Rij(zeros, randpts, Nb, Nb, Nb*ic, Nb*jc);
      D(ic,jc) = Aij;
      if (std::abs(ic - jc) <= (int64_t)admis) {
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
  gemm(A, x, Ax, 1, 0);
  print("Ida BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, D), false);

  print("Time");
  resetCounter("LR-addition");
  timing::start("BLR QR decomposition");
  for(int64_t j=0; j<Nc; j++) {
    Hierarchical Aj(Nc, 1);
    Hierarchical Qsj(Nc, 1);
    for(int64_t i=0; i<Nc; i++) {
      Aj(i, 0) = A(i, j);
      Qsj(i, 0) = A(i, j);
    }
    Hierarchical Rjj(1, 1);
    Aj.blr_col_qr(Qsj, Rjj);
    R(j, j) = std::move(Rjj(0, 0));
    //Copy column of Qsj to Q
    for(int64_t i = 0; i < Nc; i++) {
      Q(i, j) = Qsj(i, 0);
    }
    //Transpose of Qsj to be used in computing Rjk
    Hierarchical TrQsj(Qsj);
    transpose(TrQsj);
    //Process next columns
    for(int64_t k=j+1; k<Nc; k++) {
      //Take k-th column
      Hierarchical Ak(Nc, 1);
      for(int64_t i=0; i<Nc; i++) {
        Ak(i, 0) = A(i, k);
      }
      gemm(TrQsj, Ak, R(j, k), 1, 0); //Rjk = Q*j^T x A*k
      gemm(Qsj, R(j, k), Ak, -1, 1); //A*k = A*k - Q*j x Rjk
      for(int64_t i=0; i<Nc; i++) {
        A(i, k) = std::move(Ak(i, 0));
      }
    }
  }
  printCounter("LR-addition");
  timing::stopAndPrint("BLR QR decomposition", 1);

  //Residual
  Dense Rx(N);
  gemm(R, x, Rx, 1, 0);
  Dense QRx(N);
  gemm(Q, Rx, QRx, 1, 0);
  print("Residual");
  print("Rel. Error (operator norm)", l2_error(QRx, Ax), false);
  //Orthogonality
  Dense Qx(N);
  gemm(Q, x, Qx, 1, 0);
  Dense QtQx(N);
  transpose(Q);
  gemm(Q, Qx, QtQx, 1, 0);
  print("Orthogonality");
  print("Rel. Error (operator norm)", l2_error(QtQx, x), false);
  return 0;
}
