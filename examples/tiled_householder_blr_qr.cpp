#include "FRANK/FRANK.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <cassert>


using namespace FRANK;

int main(int argc, char** argv) {
  FRANK::initialize();
  const int64_t m = argc > 1 ? atoi(argv[1]) : 256;
  const int64_t n = argc > 2 ? atoi(argv[2]) : m / 2;
  const int64_t b = argc > 3 ? atoi(argv[3]) : 32;
  const double eps = argc > 4 ? atof(argv[4]) : 1e-6;
  const double admis = argc > 5 ? atof(argv[5]) : 0;
  setGlobalValue("FRANK_LRA", "rounded_addition");

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
  print("Tiled Householder BLR-QR");
  print("Time");
  timing::start("BLR-QR");
  tiled_householder_blr_qr(A, T);
  timing::stopAndPrint("BLR-QR", 1);

  //Q has same structure as A but initialized with identity
  Hierarchical Q(identity, randpts, m, n, b, eps, admis, p, q);
  left_multiply_tiled_reflector(A, T, Q, false);

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
  left_multiply_tiled_reflector(A, T, Q, true);
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
