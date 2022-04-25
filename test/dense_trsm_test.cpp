#include <cstdint>
#include <string>
#include <vector>

#include "FRANK/FRANK.h"
#include "gtest/gtest.h"

class TRSMTests : public testing::TestWithParam<int64_t> {};

TEST_P(TRSMTests, DenseTrsm) {
  int64_t n;
  n = GetParam();

  FRANK::initialize();
  const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(n)};
  FRANK::Dense A(FRANK::laplacend, randx_A, n, n);
  const FRANK::Dense x(FRANK::random_uniform, {}, n);
  FRANK::Dense b = gemm(A, x);
  FRANK::Dense L, U;

  std::tie(L, U) = FRANK::getrf(A);
  FRANK::trsm(L, b, FRANK::Mode::Lower);
  FRANK::trsm(U, b, FRANK::Mode::Upper);

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
