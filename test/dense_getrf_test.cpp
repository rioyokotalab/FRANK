#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <iostream>

#include "hicma/hicma.h"
#include "gtest/gtest.h"

class GETRFTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};

TEST_P(GETRFTests, DenseGetrf) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  //TODO getrf works only for squared matrices!!!
  if (m!=n){std::cout<<"Skipped"<<std::endl;
    return;
  }

  hicma::initialize();
  //std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?m:n)};
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m)};
  hicma::Dense A(hicma::laplacend, randx_A, m, m);

  // Set a large value on the diagonal to avoid pivoting
  int64_t d = m * n;
  int64_t n_diag = m>n?n:m;
  for (int64_t i = 0; i < n_diag; ++i) {
    A(i, i) += d--;
  }

  hicma::Dense A_copy(A);
  hicma::Dense L, U;
  std::tie(L, U) = hicma::getrf(A);
  hicma::Dense A_rebuilt = gemm(L, U);

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
