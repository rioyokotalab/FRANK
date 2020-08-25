#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>


using namespace hicma;

int main(int argc, char** argv) {
  hicma::initialize();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t Nb = argc > 2 ? atoi(argv[2]) : 32;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 16;
  double admis = argc > 4 ? atof(argv[4]) : 0;
  int64_t matCode = argc > 5 ? atoi(argv[5]) : 0;
  int64_t lra = argc > 6 ? atoi(argv[6]) : 1; updateCounter("LRA", lra);
  int64_t Nc = N / Nb;
  std::vector<std::vector<double>> randpts;
  updateCounter("LR_ADDITION_COUNTER", 1); //Enable LR addition counter

  Hierarchical A;
  Hierarchical D;
  if(matCode == 0) { //Laplace1D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    A = Hierarchical(laplacend, randpts, N, N, rank, Nb, (int64_t)admis, Nc, Nc);
    D = Hierarchical(laplacend, randpts, N, N, 0, Nb, Nc, Nc, Nc);
  } else if (matCode == 1) { //Laplace2D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    A = Hierarchical(laplacend, randpts, N, N, rank, Nb, (int64_t)admis, Nc, Nc);
    D = Hierarchical(laplacend, randpts, N, N, 0, Nb, Nc, Nc, Nc);
  } else if(matCode == 2) { //Helmholtz2D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    A = Hierarchical(helmholtznd, randpts, N, N, rank, Nb, (int64_t)admis, Nc, Nc);
    D = Hierarchical(helmholtznd, randpts, N, N, 0, Nb, Nc, Nc, Nc);
  } else { //Cauchy2D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    A = Hierarchical(cauchy2d, randpts, N, N, rank, Nb, (int64_t)admis, Nc, Nc);
    D = Hierarchical(cauchy2d, randpts, N, N, 0, Nb, Nc, Nc, Nc);
  }
  Hierarchical Q(zeros, std::vector<std::vector<double>>(), N, N, rank, Nb, (int64_t)admis, Nc, Nc);
  Hierarchical R(zeros, std::vector<std::vector<double>>(), N, N, rank, Nb, (int64_t)admis, Nc, Nc);

  print("Cond(A)", cond(Dense(A)), false);

  //For residual measurement
  Dense x(N); x = 1.0;
  Dense Ax = gemm(A, x);

  print("Ida's BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, D), false);

  print("Time");
  resetCounter("LR-addition");
  timing::start("BLR QR decomposition");
  for(int64_t j=0; j<Nc; j++) {
    orthogonalize_block_col(j, A, Q, R(j, j));
    Hierarchical QjT(Nc, 1);
    for(int64_t i=0; i<Nc; i++)
      QjT(i, 0) = Q(i, j);
    transpose(QjT);
    //Process next columns
    for(int64_t k=j+1; k<Nc; k++) {
      for(int64_t i=0; i<A.dim[0]; i++) { //Rjk = Q*j^T x A*k
        gemm(QjT(0, i), A(i, k), R(j, k), 1, 1);
      }
      for(int64_t i=0; i<A.dim[0]; i++) { //A*k = A*k - Q*j x Rjk
        gemm(Q(i, j), R(j, k), A(i, k), -1, 1);
      }
    }
  }
  timing::stopAndPrint("BLR QR decomposition", 1);
  printCounter("LR-addition");

  //Residual
  Dense Rx = gemm(R, x);
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
