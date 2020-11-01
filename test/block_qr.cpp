#include "hicma/hicma.h"

#include <cstdint>
#include <utility>
#include <vector>


using namespace hicma;

int main(int argc, char **argv) {
  hicma::initialize();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t Nb = argc > 2 ? atoi(argv[2]) : 32;
  int64_t matCode = argc > 3 ? atoi(argv[3]) : 0;
  double conditionNumber = argc > 4 ? atof(argv[4]) : 1e+0;
  int64_t Nc = N / Nb;

  Hierarchical A;
  if(matCode == 0) { //Laplace1D
    std::vector<std::vector<double>> randpts{ equallySpacedVector(N, 0.0, 1.0) };
    A = Hierarchical(laplacend, randpts, N, N, 0, Nb, Nc, Nc, Nc);
  }
  else { //Generate with LAPACK LATMS Routine
    //Configurations
    char dist = 'U'; //Uses uniform distribution when generating random SV
    std::vector<int> iseed{ 1, 23, 456, 789 };
    char sym = 'N'; //Generate symmetric or non-symmetric matrix
    double dmax = 1.0;
    int64_t kl = N-1;
    int64_t ku = N-1;
    char pack = 'N';

    std::vector<double> d(N, 0.0); //Singular values to be used
    int64_t mode = 1; //See docs
    Dense DA(N, N);
    latms(dist, iseed, sym, d, mode, conditionNumber, dmax, kl, ku, pack, DA);
    A = split(DA, Nc, Nc, true);
  }
  Hierarchical Q(zeros, std::vector<std::vector<double>>(), N, N, 0, Nb, Nc, Nc, Nc);
  Hierarchical R(zeros, std::vector<std::vector<double>>(), N, N, 0, Nb, Nc, Nc, Nc);

  print("Cond(A)", cond(Dense(A)), false);

  //For residual measurement
  Dense x(N); x = 1.0;
  Dense Ax = gemm(A, x);

  print("Block Gram Schmidt QR Decomposition");
  print("Time");
  timing::start("QR decomposition");
  qr(A, Q, R);
  timing::stopAndPrint("QR decomposition");

  //Residual
  Dense Rx = gemm(R, x);
  Dense QRx = gemm(Q, Rx);
  print("Residual");
  print("Rel. Error (operator norm)", l2_error(QRx, Ax), false);
  //Orthogonality
  Dense Qx = gemm(Q, x);
  Hierarchical Qt = transpose(Q);
  Dense QtQx = gemm(Qt, Qx);
  print("Orthogonality");
  print("Rel. Error (operator norm)", l2_error(QtQx, x), false);
  return 0;
}
