#include <cstdint>
#include <tuple>
#include <string>
#include <vector>

#include "hicma/hicma.h"
#include "gtest/gtest.h"


class RSVDTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};

TEST_P(RSVDTests, randomizedSVD) {
  int64_t n, rank;
  std::tie(n, rank) = GetParam();

  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(2*n)};
  const hicma::Dense A(hicma::laplacend, randx_A, n, n, 0, n);

  const hicma::LowRank LR(A, rank);
  const double error = hicma::l2_error(A, LR);
  EXPECT_LT(error, 1e-8);
}

INSTANTIATE_TEST_SUITE_P(
    Low_Rank, RSVDTests,
    testing::Values(std::make_tuple(2048, 16)),    
    [](const testing::TestParamInfo<RSVDTests::ParamType>& info) {
      std::string name = ("n" + std::to_string(std::get<0>(info.param)) + "rank" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });
