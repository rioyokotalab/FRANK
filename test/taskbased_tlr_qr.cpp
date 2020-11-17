#include "hicma/hicma.h"

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
  int64_t matCode = argc > 5 ? atoi(argv[5]) : 0;
  int64_t lra = argc > 6 ? atoi(argv[6]) : 1; updateCounter("LRA", lra);
  int64_t Nc = N / Nb;
  std::vector<std::vector<double>> randpts;

  Hierarchical A;
  Hierarchical D;
  if(matCode == 0) { //Laplace1D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    A = Hierarchical(laplacend, randpts, N, N, rank, Nb, admis, Nc, Nc);
    D = Hierarchical(laplacend, randpts, N, N, 0, Nb, Nc, Nc, Nc);
  } else if (matCode == 1) { //Laplace2D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    A = Hierarchical(laplacend, randpts, N, N, rank, Nb, admis, Nc, Nc);
    D = Hierarchical(laplacend, randpts, N, N, 0, Nb, Nc, Nc, Nc);
  } else if(matCode == 2) { //Helmholtz2D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    A = Hierarchical(helmholtznd, randpts, N, N, rank, Nb, admis, Nc, Nc);
    D = Hierarchical(helmholtznd, randpts, N, N, 0, Nb, Nc, Nc, Nc);
  } else { //Cauchy2D
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    randpts.push_back(equallySpacedVector(N, 0.0, 1.0));
    A = Hierarchical(cauchy2d, randpts, N, N, rank, Nb, admis, Nc, Nc);
    D = Hierarchical(cauchy2d, randpts, N, N, 0, Nb, Nc, Nc, Nc);
  }
  Hierarchical Q(identity, std::vector<std::vector<double>>(), N, N, rank, Nb, admis, Nc, Nc);
  Hierarchical T(zeros, std::vector<std::vector<double>>(), N, N, 0, Nb, Nc, Nc, Nc);

  print("Cond(A)", cond(Dense(A)), false);

  // For residual measurement
  Dense x(N); x = 1.0;
  Dense Ax = gemm(A, x);

  print("BLR QR Decomposition");
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, D), false);

  print("Time");
  double tic = get_time();
  char dep[Nc][Nc];
  // timing::start("Task-based BLR QR decomposition");
  #pragma omp parallel
  {
    #pragma omp single
    {
      for(int64_t k = 0; k < Nc; k++) {
        #pragma omp task depend(out: dep[k][k]) priority(3)
        {
          geqrt(A(k, k), T(k, k));
        }
        for(int64_t j = k+1; j < Nc; j++) {
          #pragma omp task depend(in: dep[k][k]) depend(out: dep[k][j]) priority(0)
          {
            larfb(A(k, k), T(k, k), A(k, j), true);
          }
        }
        for(int64_t i = k+1; i < Nc; i++) {
          #pragma omp task depend(in: dep[i-1][k]) depend(out: dep[i][k]) priority(2)
          {
            tpqrt(A(k, k), A(i, k), T(i, k));
          }
          for(int64_t j = k+1; j < Nc; j++) {
            #pragma omp task depend(in: dep[i][k], dep[i-1][j]) depend(out: dep[i][j]) priority(1)
            {
              tpmqrt(A(i, k), T(i, k), A(k, j), A(i, j), true);
            }
          }
        }
      }
    }
  }
  // timing::stopAndPrint("Task-based BLR QR decomposition", 1);
  double toc = get_time();
  print("Task-based BLR QR Decomposition", toc-tic);

  //Build Q: Apply Q to Id
  #pragma omp parallel
  {
    #pragma omp single
    {
      for(int64_t k = Nc-1; k >= 0; k--) {
        for(int64_t i = Nc-1; i > k; i--) {
          for(int64_t j = k; j < Nc; j++) {
            #pragma omp task depend(inout: dep[k][j], dep[i][j])
            {
              tpmqrt(A(i, k), T(i, k), Q(k, j), Q(i, j), false);
            }
          }
        }
        for(int64_t j = k; j < Nc; j++) {
          #pragma omp task depend(inout: dep[k][j])
          {
            larfb(A(k, k), T(k, k), Q(k, j), false);
          }
        }
      }
    }
  }

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
  Hierarchical Qt = transpose(Q);
  Dense QtQx = gemm(Qt, Qx);
  print("Orthogonality");
  print("Rel. Error (operator norm)", l2_error(QtQx, x), false);
  return 0;
}
