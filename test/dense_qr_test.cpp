#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <limits>

#include "FRANK/FRANK.h"
#include "gtest/gtest.h"

class QRTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};
class TruncatedQRTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, double>> {};

TEST_P(QRTests, DenseQr) {
  int64_t m, n, k;
  std::tie(m, n, k) = GetParam();

  FRANK::initialize();
  const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(m>n?m:n)};
  FRANK::Dense A(FRANK::laplacend, randx_A, m, n);
  const FRANK::Dense A_copy(A);
  FRANK::Dense Q(m, k), R(k, n);
  FRANK::qr(A, Q, R);
  const FRANK::Dense A_rebuilt = FRANK::gemm(Q, R);

  // Check accuracy
  for (int64_t i = 0; i < m; i++) {
    for (int64_t j = 0; j < n; j++) {
      EXPECT_NEAR(A_copy(i, j), A_rebuilt(i, j), 1e-12);
    }
  }
  // Check orthogonality
  const FRANK::Dense QtQ = gemm(Q, Q, 1, true, false);
  for (int64_t i = 0; i < QtQ.dim[0]; i++) {
    for (int64_t j = 0; j < QtQ.dim[1]; j++) {
      if (i == j)
        EXPECT_NEAR(QtQ(i, j), 1.0, 1e-14);
      else
        EXPECT_NEAR(QtQ(i, j), 0.0, 1e-14);
    }
  }
}

TEST_P(QRTests, DenseRq) {
  int64_t m, n, k;
  std::tie(m, n, k) = GetParam();

  FRANK::initialize();
  const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(m>n?m:n)};
  FRANK::Dense A(FRANK::laplacend, randx_A, n, m);
  const FRANK::Dense A_copy(A);
  FRANK::Dense R(n, k), Q(k, m);
  FRANK::rq(A, R, Q);
  const FRANK::Dense A_rebuilt = FRANK::gemm(R, Q);

  // Check accuracy
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < m; j++) {
      EXPECT_NEAR(A_copy(i, j), A_rebuilt(i, j), 1e-12);
    }
  }
  // Check orthogonality
  const FRANK::Dense QQt = gemm(Q, Q, 1, false, true);
  for (int64_t i = 0; i < QQt.dim[0]; i++) {
    for (int64_t j = 0; j < QQt.dim[1]; j++) {
      if (i == j)
        EXPECT_NEAR(QQt(i, j), 1.0, 1e-14);
      else
        EXPECT_NEAR(QQt(i, j), 0.0, 1e-14);
    }
  }
}

TEST_P(TruncatedQRTests, ThresholdBasedTruncation) {
  int64_t m, n;
  double eps;
  std::tie(m, n, eps) = GetParam();
  
  FRANK::initialize();
  const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(m>n?2*m:2*n)};
  
  // Construct rank deficient block
  const FRANK::Dense D(FRANK::laplacend, randx_A, m, n, 0, n);
  FRANK::Dense Q, RP;
  std::tie(Q, RP) = truncated_geqp3(D, eps);

  // Check dimensions
  EXPECT_EQ(Q.dim[0], D.dim[0]);
  EXPECT_EQ(Q.dim[1], RP.dim[0]);
  EXPECT_EQ(RP.dim[1], D.dim[1]);
  
  // Check compression error
  const double error = FRANK::l2_error(D, FRANK::gemm(Q, RP));
  EXPECT_NEAR(error, eps, 10*eps);
}

TEST_P(TruncatedQRTests, ZeroMatrixHandler) {
  int64_t m, n;
  double eps;
  std::tie(m, n, eps) = GetParam();
  
  FRANK::initialize();
  
  // Construct m x n zero matrix
  const FRANK::Dense D(m, n);
  FRANK::Dense Q, RP;
  std::tie(Q, RP) = truncated_geqp3(D, eps);
  
  // Check dimensions
  EXPECT_EQ(Q.dim[0], D.dim[0]);
  EXPECT_EQ(Q.dim[1], RP.dim[0]);
  EXPECT_EQ(RP.dim[1], D.dim[1]);
  // Ensure rank = 1
  constexpr double EPS = std::numeric_limits<double>::epsilon();
  EXPECT_EQ(Q.dim[1], 1);
  for(int64_t i = 0; i < Q.dim[0]; i++) {
    if(i == 0) {
      EXPECT_NEAR(Q(i, 0), 1.0, EPS);
    }
    else {
      EXPECT_NEAR(Q(i, 0), 0.0, EPS);
    }
  }
  for(int64_t j = 0; j < RP.dim[1]; j++) {
    EXPECT_NEAR(RP(0, j), 0.0, EPS);
  }
}

INSTANTIATE_TEST_SUITE_P(LAPACK, QRTests,
                         testing::Values(std::make_tuple(16, 16, 16),
                                         std::make_tuple(16, 8, 16),
                                         std::make_tuple(16, 8, 8),
                                         std::make_tuple(8, 16, 8)));
INSTANTIATE_TEST_SUITE_P(LAPACK, TruncatedQRTests,
                         testing::Values(std::make_tuple(32, 32, 1e-6),
                                         std::make_tuple(32, 24, 1e-6),
                                         std::make_tuple(24, 32, 1e-6),
                                         std::make_tuple(32, 32, 1e-8),
                                         std::make_tuple(32, 24, 1e-8),
                                         std::make_tuple(24, 32, 1e-8),
                                         std::make_tuple(32, 32, 1e-10),
                                         std::make_tuple(32, 24, 1e-10),
                                         std::make_tuple(24, 32, 1e-10)));
