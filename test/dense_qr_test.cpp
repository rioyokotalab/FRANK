#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "hicma/hicma.h"
#include "gtest/gtest.h"

class QRTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};

TEST_P(QRTests, DenseQr) {
  int64_t m, n, k;
  std::tie(m, n, k) = GetParam();

  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?m:n)};
  hicma::Dense A(hicma::laplacend, randx_A, m, n);
  hicma::Dense A_copy(A);
  hicma::Dense Q(m, k), R(k, n);
  hicma::qr(A, Q, R);
  hicma::Dense A_rebuilt = hicma::gemm(Q, R);

  // Check accuracy
  for (int64_t i = 0; i < m; i++) {
    for (int64_t j = 0; j < n; j++) {
      EXPECT_NEAR(A_copy(i, j), A_rebuilt(i, j), 1e-12);
    }
  }
  // Check orthogonality
  hicma::Dense QtQ = gemm(Q, Q, 1, true, false);
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

  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?m:n)};
  hicma::Dense A(hicma::laplacend, randx_A, n, m);
  hicma::Dense A_copy(A);
  hicma::Dense R(n, k), Q(k, m);
  hicma::rq(A, R, Q);
  hicma::Dense A_rebuilt = hicma::gemm(R, Q);

  // Check accuracy
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < m; j++) {
      EXPECT_NEAR(A_copy(i, j), A_rebuilt(i, j), 1e-12);
    }
  }
  // Check orthogonality
  hicma::Dense QQt = gemm(Q, Q, 1, false, true);
  for (int64_t i = 0; i < QQt.dim[0]; i++) {
    for (int64_t j = 0; j < QQt.dim[1]; j++) {
      if (i == j)
        EXPECT_NEAR(QQt(i, j), 1.0, 1e-14);
      else
        EXPECT_NEAR(QQt(i, j), 0.0, 1e-14);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(LAPACK, QRTests,
                         testing::Values(std::make_tuple(16, 16, 16),
                                         std::make_tuple(16, 8, 16),
                                         std::make_tuple(16, 8, 8),
                                         std::make_tuple(8, 16, 8)));
