#include "hicma/hicma.h"

#include "yorel/yomm2/cute.hpp"

#include <cstdint>
#include <utility>
#include <vector>


using namespace hicma;

int main(int argc, char** argv) {
  yorel::yomm2::update_methods();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t Nb = argc > 2 ? atoi(argv[2]) : 32;
  int64_t matCode = argc > 3 ? atoi(argv[3]) : 0;
  double conditionNumber = argc > 4 ? atof(argv[4]) : 1e+0;
  int64_t Nc = N / Nb;
  std::vector<std::vector<double>> randpts;
  Dense DA;

  if(matCode == 0 || matCode == 1) { //Laplace1D or Helmholtz1D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  }
  else if(matCode == 2) { //Ill-conditioned Cauchy2D (Matrix A3 From HODLR Paper)
    randpts.push_back(equallySpacedVector(N, -1.25, 998.25));
    randpts.push_back(equallySpacedVector(N, -0.15, 999.45));
  }
  else { //Ill-conditioned generated from DLATMS
    //Configurations
    char dist = 'U'; //Uses uniform distribution when generating random SV
    std::vector<int> iseed{ 1, 23, 456, 789 };
    char sym = 'N'; //Generate symmetric matrix
    double dmax = 1.0;
    int64_t kl = N-1;
    int64_t ku = N-1;
    char pack = 'N';

    std::vector<double> d(N, 0.0); //Singular values to be used
    int64_t mode = 1; //See docs
    Dense _DA(N, N);
    latms(dist, iseed, sym, d, mode, conditionNumber, dmax, kl, ku, pack, _DA);
    DA = std::move(_DA);

    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  }

  Hierarchical A(Nc, Nc);
  Hierarchical T(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  for(int64_t ic = 0; ic < Nc; ic++) {
    for(int64_t jc = 0; jc < Nc; jc++) {
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
      else {
        Dense _Aij(zeros, randpts[0], Nb, Nb, Nb*ic, Nb*jc);
        getSubmatrix(DA, Nb, Nb, Nb*ic, Nb*jc, _Aij);
        Aij = std::move(_Aij);
      }
      Dense Qij(identity, randpts[0], Nb, Nb, Nb*ic, Nb*jc);
      Dense Tij(zeros, randpts[0], Nb, Nb);
      A(ic, jc) = Aij;
      Q(ic, jc) = Qij;
      T(ic, jc) = Tij;
    }
  }
  //Compute condition number of A
  print("Cond(A)", cond(Dense(A)), false);

  // For residual measurement
  Dense x(N); x = 1.0;
  Dense Ax = gemm(A, x);
  print("Time");
  timing::start("QR decomposition");
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
  timing::stopAndPrint("QR decomposition");
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
  //Build R: Take upper triangular part of modified A
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<=i; j++) {
      if(i == j)
        zero_lowtri(A(i, j));
      else
        zero_whole(A(i, j));
    }
  }
  //Residual
  Dense Rx = gemm(A, x);
  Dense QRx = gemm(Q, Rx);
  print("Accuracy");
  print("Rel. Error (operator norm)", l2_error(QRx, Ax), false);
  //Orthogonality
  Dense Qx = gemm(Q, x);
  transpose(Q);
  Dense QtQx = gemm(Q, Qx);
  print("Orthogonality");
  print("Rel. Error (operator norm)", l2_error(QtQx, x), false);
  return 0;
}
