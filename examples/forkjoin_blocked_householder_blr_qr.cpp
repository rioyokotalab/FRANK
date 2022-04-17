#include "hicma/hicma.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <omp.h>
#include <sys/time.h>
#include <cstdlib>
#include <cassert>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

using namespace hicma;

int main(int argc, char** argv) {
  hicma::initialize();
  const int64_t m = argc > 1 ? atoi(argv[1]) : 256;
  const int64_t n = argc > 2 ? atoi(argv[2]) : m / 2;
  const int64_t b = argc > 3 ? atoi(argv[3]) : 32;
  const double eps = argc > 4 ? atof(argv[4]) : 1e-6;
  const double admis = argc > 5 ? atof(argv[5]) : 0;
  setGlobalValue("HICMA_LRA", "rounded_addition");
  setGlobalValue("HICMA_DISABLE_TIMER", "1");

  assert(m >= n);
  assert(m % b == 0);
  assert(n % b == 0);
  const int64_t p = m / b;
  const int64_t q = n / b;

  const std::vector<std::vector<double>> randpts {
    equallySpacedVector(m, 0.0, 1.0),
    equallySpacedVector(m, 0.0, 1.0)
  };
  const Hierarchical D(laplacend, randpts, m, n, b, b, p, p, q);
  Hierarchical A(laplacend, randpts, m, n, b, eps, admis, p, q);
  print("BLR Compression Accuracy");
  print("Rel. L2 Error", l2_error(D, A), false);

  Hierarchical T(q, 1);
  print("Forkjoin Blocked Householder BLR-QR");
  print("Number of Threads: ", omp_get_max_threads());
  print("Time");
  const double tic = get_time();
  for(int64_t k = 0; k < q; k++) {
    triangularize_block_col(k, A, T);
    #pragma omp parallel for schedule(dynamic)
    for(int j = k+1; j < q; j++) {
      apply_block_col_householder(A, T, k, true, A, j);
    }
  }
  const double toc = get_time();
  print("BLR-QR", toc-tic);

  //Q has same structure as A but initialized with identity
  Hierarchical Q(identity, randpts, m, n, b, eps, admis, p, q);
  for(int64_t k = q-1; k >= 0; k--) {
    #pragma omp parallel for schedule(dynamic)
    for(int j = k; j < q; j++) {
      apply_block_col_householder(A, T, k, false, Q, j);
    }
  }

  print("BLR-QR Accuracy");
  //Residual
  Hierarchical QR(Q);
  //R is taken from upper triangular part of A
  Hierarchical R(q, q);
  for(int64_t i=0; i<q; i++) {
    for(int64_t j=i; j<q; j++) {
      R(i, j) = A(i, j);
    }
  }
  //Use trmm here since lower triangular part of R is not initialized
  trmm(R, QR, Side::Right, Mode::Upper, 'n', 'n', 1.);
  print("Residual", l2_error(D, QR), false);
  
  //Orthogonality
  //QtQ: Left multiply Q^T to Q
  for(int64_t k = 0; k < q; k++) {
    #pragma omp parallel for schedule(dynamic)
    for(int64_t j = k; j < q; j++) {
      apply_block_col_householder(A, T, k, true, Q, j);
    }
  }
  //Take square part as Q^T x Q (assuming m >= n)
  Hierarchical QtQ(q, q);
  for(int64_t i = 0; i < q; i++) {
    for(int64_t j = 0; j < q; j++) {
      QtQ(i, j) = Q(i, j);
    }
  }
  print("Orthogonality", l2_error(Dense(identity, {}, n, n), QtQ), false);
  return 0;
}
