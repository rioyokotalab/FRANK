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
  Hierarchical Q(Nc, Nc);
  Hierarchical R(zeros, std::vector<std::vector<double>>(), N, N, 0, Nb, Nc, Nc, Nc);

  print("Cond(A)", cond(Dense(A)), false);

  //For residual measurement
  Dense x(N); x = 1.0;
  Dense Ax = gemm(A, x);

  print("Block Gram Schmidt QR Decomposition");
  print("Time");
  timing::start("QR decomposition");
  for(int64_t j = 0; j < Nc; j++) {
    Hierarchical HAsj(Nc, 1);
    for(int64_t i = 0; i < Nc; i++) {
      HAsj(i, 0) = A(i, j);
    }
    Dense DAsj(HAsj);
    Dense DQsj(DAsj.dim[0], DAsj.dim[1]);
    Dense Rjj(Nb, Nb);
    qr(DAsj, DQsj, Rjj); //[Q*j, Rjj] = QR(A*j)
    R(j, j) = std::move(Rjj);
    //Move Dense Qsj to Hierarchical Q
    Hierarchical HQsj = split(DQsj, Nc, 1, true);
    for(int64_t i = 0; i < Nc; i++) {
      Q(i, j) = std::move(HQsj(i, 0));
    }
    //Process next columns
    for(int64_t k = j + 1; k < Nc; k++) {
      //Take k-th column
      Hierarchical HAsk(Nc, 1);
      for(int64_t i = 0; i < Nc; i++) {
        HAsk(i, 0) = A(i, k);
      }
      Dense DAsk(HAsk);
      Dense DRjk(Nb, Nb);
      gemm(DQsj, DAsk, DRjk, 1, 1, true, false); //Rjk = Qsj^T x Ask
      gemm(DQsj, DRjk, DAsk, -1, 1); //A*k = A*k - Q*j x Rjk
      R(j, k) = std::move(DRjk);
      Hierarchical _HAsk = split(DAsk, Nc, 1, true);
      for(int64_t i = 0; i < Nc; i++) {
        A(i, k) = std::move(_HAsk(i, 0));
      }
    }
  }
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
