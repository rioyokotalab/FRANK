#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <iostream>

#include "FRANK/FRANK.h"
#include "gtest/gtest.h"

class GETRFTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};

TEST_P(GETRFTests, DenseGetrf) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  //TODO getrf works only for squared matrices!!!
  if (m!=n){std::cout<<"Skipped"<<std::endl;
    return;
  }

  FRANK::initialize();
  const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(m)};
  FRANK::Dense A(FRANK::laplacend, randx_A, m, m);

  // Set a large value on the diagonal to avoid pivoting
  int64_t d = m * n;
  const int64_t n_diag = m>n?n:m;
  for (int64_t i = 0; i < n_diag; ++i) {
    A(i, i) += d--;
  }

  const FRANK::Dense A_copy(A);
  FRANK::Dense L, U;
  std::tie(L, U) = FRANK::getrf(A);
  const FRANK::Dense A_rebuilt = gemm(L, U);

  // Check result
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_DOUBLE_EQ(A_rebuilt(i, j), A_copy(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, GETRFTests,
    testing::Combine(testing::Values(8, 16, 32), testing::Values(8, 16, 32)),
    [](const testing::TestParamInfo<GETRFTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)) + "n" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });
