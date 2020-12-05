#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <omp.h>
#include <sys/time.h>
#include <cstdlib>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

using namespace hicma;

int main(int argc, char** argv) {
  hicma::initialize();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t Nb = argc > 2 ? atoi(argv[2]) : 32;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 16;
  double admis = argc > 4 ? atof(argv[4]) : 0;
  int64_t Nc = N / Nb;
  setGlobalValue("HICMA_LRA", "rounded_orth");
  setGlobalValue("HICMA_DISABLE_TIMER", "1");

  std::vector<std::vector<double>> randpts;
  randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
  Hierarchical D(laplacend, randpts, N, N, Nb, Nb, Nc, Nc, Nc);
  Hierarchical A(laplacend, randpts, N, N, rank, Nb, admis, Nc, Nc);
  print("BLR Compression");
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, D), false);

  Hierarchical Q(zeros, std::vector<std::vector<double>>(), N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical R(zeros, std::vector<std::vector<double>>(), N, N, rank, Nb, admis, Nc, Nc);
  //For residual measurement
  Dense x(N); x = 1.0;
  Dense Ax = gemm(A, x);

  print("Forkjoin Ida's BLR QR Decomposition");
  print("Time");
  double tic = get_time();
  for (int64_t j=0; j<A.dim[1]; j++) {
    orthogonalize_block_col(j, A, Q, R(j, j));
    Hierarchical QjT(1, Q.dim[0]);
    for (int64_t i=0; i<Q.dim[0]; i++) {
      QjT(0, i) = transpose(Q(i, j));
    }
    for (int64_t k=j+1; k<A.dim[1]; k++) {
      for(int64_t i=0; i<A.dim[0]; i++) { //Rjk = Q*j^T x A*k
        gemm(QjT(0, i), A(i, k), R(j, k), 1, 1);
      }
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t k=j+1; k<A.dim[1]; k++)
      for(int64_t i=0; i<A.dim[0]; i++) //A*k = A*k - Q*j x Rjk
        gemm(Q(i, j), R(j, k), A(i, k), -1, 1);
  }
  double toc = get_time();
  print("BLR QR Decomposition", toc-tic);

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
