#include "FRANK/FRANK.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <omp.h>
#include <sys/time.h>
#include <cstdlib>
#include <cassert>

using namespace FRANK;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

int main(int argc, char** argv) {
  FRANK::initialize();
  const int64_t m = argc > 1 ? atoi(argv[1]) : 256;
  const int64_t n = argc > 2 ? atoi(argv[2]) : m / 2;
  const int64_t b = argc > 3 ? atoi(argv[3]) : 32;
  const double eps = argc > 4 ? atof(argv[4]) : 1e-6;
  const double admis = argc > 5 ? atof(argv[5]) : 0;
  setGlobalValue("FRANK_LRA", "rounded_addition");
  setGlobalValue("FRANK_DISABLE_TIMER", "1");

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

  //Q and R have same structure as A but initially contain zeros
  Hierarchical Q(zeros, randpts, m, n, b, eps, admis, p, q);
  Hierarchical R(zeros, randpts, n, n, b, eps, admis, q, q);
  print("Forkjoin Blocked Modified Gram-Schmidt BLR-QR");
  print("Number of Threads: ", omp_get_max_threads());
  print("Time");
  const double tic = get_time();
  for (int64_t j=0; j<q; j++) {
    orthogonalize_block_col(j, A, Q, R(j, j));
    Hierarchical QjT(1, p);
    for (int64_t i=0; i<p; i++) {
      QjT(0, i) = transpose(Q(i, j));
    }
    #pragma omp parallel for schedule(dynamic)
    for (int64_t k=j+1; k<q; k++) {
      for(int64_t i=0; i<p; i++) { //Rjk = Q*j^T x A*k
        gemm(QjT(0, i), A(i, k), R(j, k), 1, 1);
      }
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t k=j+1; k<q; k++) {
      for(int64_t i=0; i<p; i++) { //A*k = A*k - Q*j x Rjk
        gemm(Q(i, j), R(j, k), A(i, k), -1, 1);
      }
    }
  }
  const double toc = get_time();
  print("BLR-QR", toc-tic);

  print("BLR-QR Accuracy");
  //Residual
  Hierarchical QR(Q);
  trmm(R, QR, FRANK::Side::Right, FRANK::Mode::Upper, 'n', 'n', 1.);
  print("Residual", l2_error(D, QR), false);
  //Orthogonality
  Hierarchical QtQ(zeros, randpts, n, n, b, eps, admis, q, q);
  const Hierarchical Qt = transpose(Q);
  gemm(Qt, Q, QtQ, 1, 0);
  print("Orthogonality", l2_error(Dense(identity, {}, n, n), QtQ), false);
  return 0;
}
