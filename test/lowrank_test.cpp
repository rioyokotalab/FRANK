#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include <iostream>

#include "hicma/hicma.h"
#include "gtest/gtest.h"

class LowRankTest
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, double>> {};

TEST_P(LowRankTest, ConstructionByThreshold) {
  int64_t m, n;
  double eps;
  std::tie(m, n, eps) = GetParam();
  
  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?2*m:2*n)};
  
  // Construct rank deficient block
  hicma::Dense D(hicma::laplacend, randx_A, m, n, 0, n);
  hicma::LowRank A(D, eps);
  // Check compression error
  double error = hicma::l2_error(D, A);
  EXPECT_NEAR(error, eps, 10*eps);
}

INSTANTIATE_TEST_SUITE_P(LowRank, LowRankTest,
                         testing::Values(std::make_tuple(32, 32, 1e-6),
                                         std::make_tuple(32, 24, 1e-6),
                                         std::make_tuple(24, 32, 1e-6),
					 std::make_tuple(32, 32, 1e-8),
                                         std::make_tuple(32, 24, 1e-8),
                                         std::make_tuple(24, 32, 1e-8),
					 std::make_tuple(32, 32, 1e-10),
                                         std::make_tuple(32, 24, 1e-10),
                                         std::make_tuple(24, 32, 1e-10),
					 std::make_tuple(32, 32, 1e-12),
                                         std::make_tuple(32, 24, 1e-12),
                                         std::make_tuple(24, 32, 1e-12)));
