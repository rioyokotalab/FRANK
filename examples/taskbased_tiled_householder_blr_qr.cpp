#include "hicma/hicma.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <omp.h>
#include <sys/time.h>
#include <cstdlib>
#include <cassert>

using namespace hicma;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

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

  Hierarchical T(p, q);
  for(int64_t i = 0; i < p; i++) {
    for(int64_t j = 0; j < q; j++) {
      T(i, j) = Dense(i < j ? 0 : b, i < j ? 0 : b);
    }
  }
  print("Taskbased Tiled Householder BLR-QR");
  print("Number of Threads: ", omp_get_max_threads());
  print("Time");
  const double tic = get_time();
  [[maybe_unused]] char dep[p][q];
  #pragma omp parallel
  {
    #pragma omp single
    {
      for(int64_t k = 0; k < q; k++) {
        #pragma omp task depend(out: dep[k][k]) priority(3)
        {
          geqrt(A(k, k), T(k, k));
        }
        for(int64_t j = k+1; j < q; j++) {
          #pragma omp task depend(in: dep[k][k]) depend(out: dep[k][j]) priority(0)
          {
            larfb(A(k, k), T(k, k), A(k, j), true);
          }
        }
        for(int64_t i = k+1; i < p; i++) {
          #pragma omp task depend(in: dep[i-1][k]) depend(out: dep[i][k]) priority(2)
          {
            tpqrt(A(k, k), A(i, k), T(i, k));
          }
          for(int64_t j = k+1; j < q; j++) {
            #pragma omp task depend(in: dep[i][k], dep[i-1][j]) depend(out: dep[i][j]) priority(1)
            {
              tpmqrt(A(i, k), T(i, k), A(k, j), A(i, j), true);
            }
          }
        }
      }
    }
  }
  const double toc = get_time();
  print("BLR-QR", toc-tic);

  //Q has same structure as A but initialized with identity
  Hierarchical Q(identity, randpts, m, n, b, eps, admis, p, q);
  #pragma omp parallel
  {
    #pragma omp single
    {
      for(int64_t k = q-1; k >= 0; k--) {
        for(int64_t i = p-1; i > k; i--) {
          for(int64_t j = k; j < q; j++) {
            #pragma omp task depend(inout: dep[k][j], dep[i][j])
            {
              tpmqrt(A(i, k), T(i, k), Q(k, j), Q(i, j), false);
            }
          }
        }
        for(int64_t j = k; j < q; j++) {
          #pragma omp task depend(inout: dep[k][j])
          {
            larfb(A(k, k), T(k, k), Q(k, j), false);
          }
        }
      }
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
  #pragma omp parallel
  {
    #pragma omp single
    {
      for(int64_t k = 0; k < q; k++) {
        for(int64_t j = k; j < q; j++) {
          #pragma omp task depend(in: dep[k][k]) depend(out: dep[k][j]) priority(0)
          {
            larfb(A(k, k), T(k, k), Q(k, j), true);
          }
        }
        for(int64_t i = k+1; i < p; i++) {
          for(int64_t j = k; j < q; j++) {
            #pragma omp task depend(in: dep[i][k], dep[i-1][j]) depend(out: dep[i][j]) priority(1)
            {
              tpmqrt(A(i, k), T(i, k), Q(k, j), Q(i, j), true);
            }
          }
        }
      }
    }
  }
  // Take square part as Q^T x Q (assuming m >= n)
  Hierarchical QtQ(q, q);
  for(int64_t i = 0; i < q; i++) {
    for(int64_t j = 0; j < q; j++) {
      QtQ(i, j) = Q(i, j);
    }
  }
  print("Orthogonality", l2_error(Dense(identity, {}, n, n), QtQ), false);
  return 0;
}
