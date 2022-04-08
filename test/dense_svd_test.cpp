#include <tuple>
#include <string>
#include <vector>

#include "hicma/hicma.h"
#include "gtest/gtest.h"

class SVDTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};

TEST_P(SVDTests, DenseSvd) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(n)};
  
  // double precision
  hicma::Dense A(hicma::laplacend, randx_A, n, n);
  hicma::Dense A_copy(A);
  hicma::Dense U, S, V;
  std::tie(U, S, V) = hicma::svd(A);
  hicma::Dense A_rebuilt = hicma::gemm(hicma::gemm(U, S), V);

  // Check result
  for (int64_t i = 0; i < A.dim[0]; ++i) {
    for (int64_t j = 0; j < A.dim[1]; ++j) {
      EXPECT_NEAR(A_rebuilt(i, j), A_copy(i, j), 1e-11);
    }
  }

  // single precision
  hicma::Dense<float> Af(hicma::laplacend, randx_A, n, n);
  hicma::Dense<float> Af_copy(Af);
  hicma::Dense<float> Uf, Sf, Vf;
  std::tie(Uf, Sf, Vf) = hicma::svd(Af);
  hicma::Dense<float> Af_rebuilt = hicma::gemm(hicma::gemm(Uf, Sf), Vf);

  // Check result
  for (int64_t i = 0; i < Af.dim[0]; ++i) {
    for (int64_t j = 0; j < Af.dim[1]; ++j) {
      EXPECT_NEAR(Af_rebuilt(i, j), Af_copy(i, j), 1e-3);
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
