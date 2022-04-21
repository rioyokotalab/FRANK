#include <tuple>
#include <string>
#include <vector>

#include "FRANK/FRANK.h"
#include "gtest/gtest.h"

class SVDTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};

TEST_P(SVDTests, DenseSvd) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  FRANK::initialize();
  const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(n)};
  FRANK::Dense A(FRANK::laplacend, randx_A, n, n);

  const FRANK::Dense A_copy(A);
  FRANK::Dense U, S, V;
  std::tie(U, S, V) = FRANK::svd(A);
  const FRANK::Dense A_rebuilt = FRANK::gemm(FRANK::gemm(U, S), V);

  // Check result
  for (int64_t i = 0; i < A.dim[0]; ++i) {
    for (int64_t j = 0; j < A.dim[1]; ++j) {
      EXPECT_NEAR(A_rebuilt(i, j), A_copy(i, j), 1e-11);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, SVDTests,
    testing::Combine(testing::Values(8, 16), testing::Values(8, 16)),
    [](const testing::TestParamInfo<SVDTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)) + "n" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });
