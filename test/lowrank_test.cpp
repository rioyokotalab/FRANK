#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include <iostream>

#include "FRANK/FRANK.h"
#include "gtest/gtest.h"

class LowRankTest_FixedRank
  : public testing::TestWithParam<std::tuple<std::string, int64_t, int64_t, int64_t>> {};
class LowRankTest_FixedAccuracy
  : public testing::TestWithParam<std::tuple<std::string, int64_t, int64_t, double>> {};

TEST_P(LowRankTest_FixedRank, Construction) {
  std::string lr_add_alg;
  int64_t m, n, rank;
  std::tie(lr_add_alg, m, n, rank) = GetParam();

  FRANK::initialize();
  FRANK::setGlobalValue("FRANK_LRA", lr_add_alg);
  const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(m>n?2*m:2*n)};

  // Construct rank deficient block
  const FRANK::Dense D(FRANK::laplacend, randx_A, m, n, 0, n);
  const FRANK::LowRank A(D, rank);
  // Check rank
  EXPECT_EQ(A.rank, rank);
}

TEST_P(LowRankTest_FixedRank, Addition) {
  std::string lr_add_alg;
  int64_t m, n, rank;
  std::tie(lr_add_alg, m, n, rank) = GetParam();

  FRANK::initialize();
  // Set low-rank addition algorithm
  FRANK::setGlobalValue("FRANK_LRA", lr_add_alg);
  const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(m>n?2*m:2*n)};

  const FRANK::Dense DA(FRANK::laplacend, randx_A, m, n, 0, n);
  FRANK::LowRank A(DA, rank);

  // Add with similar low-rank block
  const FRANK::Dense DB(DA);
  const FRANK::LowRank B(DB, rank);
  A += B;
  EXPECT_EQ(A.rank, rank);

  // Add with low-rank block with different rank
  const FRANK::Dense DC(DA);
  const FRANK::LowRank C(DC, std::min(rank + 4, std::min(m, n)));
  A += C;
  EXPECT_EQ(A.rank, rank);

  // Add with full-rank random block
  const FRANK::Dense DD(FRANK::random_normal, randx_A, m, n);
  const FRANK::LowRank D(DD, rank);
  A += D;
  EXPECT_EQ(A.rank, rank);
}

TEST_P(LowRankTest_FixedAccuracy, Construction) {
  std::string lr_add_alg;
  int64_t m, n;
  double eps;
  std::tie(lr_add_alg, m, n, eps) = GetParam();

  FRANK::initialize();
  FRANK::setGlobalValue("FRANK_LRA", lr_add_alg);
  const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(m>n?2*m:2*n)};

  // Construct rank deficient block
  const FRANK::Dense D(FRANK::laplacend, randx_A, m, n, 0, n);
  const FRANK::LowRank A(D, eps);
  // Check compression error
  const double error = FRANK::l2_error(D, A);
  EXPECT_NEAR(error, eps, 10*eps);
}

TEST_P(LowRankTest_FixedAccuracy, Addition) {
  std::string lr_add_alg;
  int64_t m, n;
  double eps, error;
  std::tie(lr_add_alg, m, n, eps) = GetParam();

  FRANK::initialize();
  // Set low-rank addition algorithm
  FRANK::setGlobalValue("FRANK_LRA", lr_add_alg);
  const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(m>n?2*m:2*n)};

  const FRANK::Dense DA(FRANK::laplacend, randx_A, m, n, 0, n);
  FRANK::LowRank A(DA, eps);

  // Add with similar low-rank block
  const FRANK::Dense DB(DA);
  const FRANK::LowRank B(DB, eps);
  A += B;
  error = FRANK::l2_error((DA+DB), A);
  EXPECT_NEAR(error, eps, 10*eps);

  // Add with low-rank block with different rank
  const FRANK::Dense DC(DA);
  const FRANK::LowRank C(DC, eps*1e-2);
  A += C;
  error = FRANK::l2_error((DA+DB+DC), A);
  EXPECT_NEAR(error, eps, 10*eps);

  // Add with full-rank random block
  const FRANK::Dense DD(FRANK::random_normal, randx_A, m, n);
  const FRANK::LowRank D(DD, eps);
  A += D;
  error = FRANK::l2_error((DA+DB+DC+DD), A);
  EXPECT_NEAR(error, eps, 10*eps);
}

INSTANTIATE_TEST_SUITE_P(LowRank, LowRankTest_FixedRank,
                         testing::Combine(testing::Values("rounded_addition", "fast_rounded_addition"),
                                          testing::Values(64, 32),
                                          testing::Values(64, 21),
                                          testing::Values(1, 2, 4, 8)
                                          ));

INSTANTIATE_TEST_SUITE_P(LowRank, LowRankTest_FixedAccuracy,
                         testing::Combine(testing::Values("rounded_addition", "fast_rounded_addition"),
                                          testing::Values(64, 32),
                                          testing::Values(64, 32),
                                          testing::Values(1e-6, 1e-8, 1e-10, 1e-12)
                                          ));
