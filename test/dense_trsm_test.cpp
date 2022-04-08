#include <cstdint>
#include <string>
#include <vector>

#include "hicma/hicma.h"
#include "gtest/gtest.h"

class TRSMTests : public testing::TestWithParam<int64_t> {};

TEST_P(TRSMTests, DenseTrsm) {
  int64_t n;
  n = GetParam();

  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(n)};
  hicma::Dense A(hicma::laplacend, randx_A, n, n);
  hicma::Dense x(hicma::random_uniform, n);
  hicma::Dense b = gemm(A, x);
  hicma::Dense L, U;

  std::tie(L, U) = hicma::getrf(A);
  hicma::trsm(L, b, hicma::TRSM_LOWER);
  hicma::trsm(U, b, hicma::TRSM_UPPER);

  // Check result
  for (int64_t i = 0; i < n; ++i) {
    EXPECT_NEAR(x(i, 0), b(i, 0), 1e-12);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BLAS, TRSMTests,
    testing::Values(8, 16, 32),
    [](const testing::TestParamInfo<TRSMTests::ParamType>& info) {
      std::string name = ("n" + std::to_string(info.param));
      return name;
    });
