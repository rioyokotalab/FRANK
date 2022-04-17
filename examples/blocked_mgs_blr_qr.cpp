#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <cassert>


using namespace hicma;

int main(int argc, char** argv) {
  hicma::initialize();
  const int64_t m = argc > 1 ? atoi(argv[1]) : 256;
  const int64_t n = argc > 2 ? atoi(argv[2]) : m / 2;
  const int64_t b = argc > 3 ? atoi(argv[3]) : 32;
  const double eps = argc > 4 ? atof(argv[4]) : 1e-6;
  const double admis = argc > 5 ? atof(argv[5]) : 0;
  setGlobalValue("HICMA_LRA", "rounded_addition");

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
  print("Blocked Modified Gram-Schmidt BLR-QR");
  print("Time");
  timing::start("BLR-QR");
  mgs_qr(A, Q, R);
  timing::stopAndPrint("BLR-QR", 1);

  print("BLR-QR Accuracy");
  //Residual
  Hierarchical QR(Q);
  trmm(R, QR, hicma::Side::Right, hicma::Mode::Upper, 'n', 'n', 1.);
  print("Residual", l2_error(D, QR), false);
  //Orthogonality
  Hierarchical QtQ(zeros, randpts, n, n, b, eps, admis, q, q);
  const Hierarchical Qt = transpose(Q);
  gemm(Qt, Q, QtQ, 1, 0);
  print("Orthogonality", l2_error(Dense(identity, {}, n, n), QtQ), false);
  return 0;
}
