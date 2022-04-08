#include <cstdint>
#include <tuple>
#include <string>
#include <vector>

#include "hicma/hicma.h"
#include "gtest/gtest.h"


using namespace hicma;

class RSVDTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};

// TODO this is actually a low rank test and has been moved there. Replacement?
TEST_P(RSVDTests, randomizedSVD) {
  int64_t n, rank;
  std::tie(n, rank) = GetParam();

  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(2*n)};
  hicma::Dense A(hicma::laplacend, randx_A, n, n, 0, n);

  LowRank LR(A, rank);
  double error = l2_error(A, LR);
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
