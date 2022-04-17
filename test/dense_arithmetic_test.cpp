#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "hicma/hicma.h"
#include "gtest/gtest.h"

class ArithmeticTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};
//class MatMulOperatorTests
//    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};
class ArithmeticTests2
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, double>> {};


TEST_P(ArithmeticTests, DenseAddition) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?m:n)};
  const hicma::Dense A(hicma::laplacend, randx_A, m, n);
  const std::vector<std::vector<double>> randx_B{hicma::get_sorted_random_vector(m>n?m:n)};
  const hicma::Dense B(hicma::laplacend, randx_B, m, n);
  const hicma::Dense C = A + B;

  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_EQ(C(i, j), A(i, j) + B(i, j));
    }
}

TEST_P(ArithmeticTests, DensePlusEquals) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?m:n)};
  hicma::Dense A(hicma::laplacend, randx_A, m, n);
  const hicma::Dense A_check(A);
  const std::vector<std::vector<double>> randx_B{hicma::get_sorted_random_vector(m>n?m:n)};
  const hicma::Dense B(hicma::laplacend, randx_B, m, n);
  A += B;

  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_EQ(A_check(i, j) + B(i, j), A(i, j));
    }
}

TEST_P(ArithmeticTests, DenseSubtraction) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?m:n)};
  const hicma::Dense A(hicma::laplacend, randx_A, m, n);
  const std::vector<std::vector<double>> randx_B{hicma::get_sorted_random_vector(m>n?m:n)};
  const hicma::Dense B(hicma::laplacend, randx_B, m, n);
  const hicma::Dense C = A - B;

  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_EQ(C(i, j), A(i, j) - B(i, j));
    }
}
/*
//TODO implement
TEST_P(ArithmeticTests, MinusEqualsOperator) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?m:n)};
  hicma::Dense A(hicma::laplacend, randx_A, m, n);
  const hicma::Dense A_check(A);
  const std::vector<std::vector<double>> randx_B{hicma::get_sorted_random_vector(m>n?m:n)};
  const hicma::Dense B(hicma::laplacend, randx_B, m, n);
  A -= B;

  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_EQ(A_check(i, j) - B(i, j), A(i, j));
    }
}

//TODO implement
TEST_P(ArithmeticTests, abs) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?m:n)};
  const hicma::Dense A(hicma::laplacend, randx_A, m, n);
  const hicma::Dense A_check = abs(A);

  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_EQ(A_check(i, j), A(i, j) < 0 ? -A(i, j) : A(i, j));
    }
}
*/

TEST_P(ArithmeticTests, Transpose) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?m:n)};
  const hicma::Dense A(hicma::laplacend, randx_A, m, n);
  const hicma::Dense A_copy(A);
  const hicma::Dense A_trans = transpose(A);

  EXPECT_EQ(A_trans.dim[0], n);
  EXPECT_EQ(A_trans.dim[1], m);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_EQ(A(i, j), A_trans(j, i));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, ArithmeticTests,
    testing::Values(std::make_tuple(50, 50), std::make_tuple(23, 75),
                    std::make_tuple(100, 66)),
    [](const testing::TestParamInfo<ArithmeticTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)) + "n" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });

TEST_P(ArithmeticTests2, DenseScalarMultiplicationEqualsOperator) {
  int64_t m, n;
  double alpha;
  std::tie(m, n, alpha) = GetParam();

  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>n?m:n)};
  hicma::Dense A(hicma::laplacend, randx_A, m, n);
  const hicma::Dense A_copy(A);
  A *= alpha;

  // Check result
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_EQ(A_copy(i, j) * alpha, A(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Operator, ArithmeticTests2,
    testing::Values(std::make_tuple(5, 5, 7.9834), std::make_tuple(11, 21, -4),
                    std::make_tuple(18, 5, 1 / 8)),
    [](const testing::TestParamInfo<ArithmeticTests2::ParamType>& info) {
      std::string name = ("M" + std::to_string(std::get<0>(info.param)) + "N" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });

/*
// TODO implement
TEST_P(MatMulOperatorTests, MultiplicationOperator) {
  int64_t m, n, k;
  std::tie(m, n, k) = GetParam();

  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>k?m:k)};
  const hicma::Dense A(hicma::laplacend, randx_A, m, k);
  const std::vector<std::vector<double>> randx_B{hicma::get_sorted_random_vector(k>n?k:n)};
  const hicma::Dense B(hicma::laplacend, randx_B, k, n);
  const hicma::Dense C_check = A * B;
  hicma::Dense C(m, n);
  hicma::gemm(A, B, C, false, false, 1, 0);

  // Check result
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_FLOAT_EQ(C_check(i, j), C(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Operator, MatMulOperatorTests,
    testing::Combine(testing::Values(16, 32, 64), testing::Values(16, 32, 64),
                     testing::Values(16, 32, 64)),
    [](const testing::TestParamInfo<MatMulOperatorTests::ParamType>& info) {
      std::string name = ("M" + std::to_string(std::get<0>(info.param)) + "K" +
                          std::to_string(std::get<1>(info.param)) + "N" +
                          std::to_string(std::get<2>(info.param)));
      return name;
    });
*/
