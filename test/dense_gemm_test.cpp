#include <cstdint>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <vector>
#include <string>

#include "hicma/hicma.h"
#include "gtest/gtest.h"

void naive_gemm(const hicma::Dense &A, const hicma::Dense &B, hicma::Dense &C,
                const bool transA, const bool transB, const double alpha, const double beta) {
  for (int64_t i = 0; i < C.dim[0]; ++i) {
    for (int64_t j = 0; j < C.dim[1]; ++j) {
      C(i, j) =
          (beta * C(i, j) +
           alpha * (transA ? A(0, i) : A(i, 0)) * (transB ? B(j, 0) : B(0, j)));
      for (int64_t k = 1; k < (transA?A.dim[0]:A.dim[1]); ++k) {
        C(i, j) += (alpha * (transA ? A(k, i) : A(i, k)) *
                          (transB ? B(j, k) : B(k, j)));
      }
    }
  }
}

class GEMMTests
    : public testing::TestWithParam<
          std::tuple<int64_t, int64_t, int64_t, bool, bool, double, double> > {
};

TEST_P(GEMMTests, DenseGemm) {
  int64_t m, n, k;
  bool transA, transB;
  double alpha, beta;
  std::tie(m, k, n, transA, transB, alpha, beta) = GetParam();
  
  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>k?m:k)};
  const hicma::Dense A(hicma::laplacend, randx_A, transA?k:m, transA?m:k);
  const std::vector<std::vector<double>> randx_B{hicma::get_sorted_random_vector(k>n?k:n)};
  const hicma::Dense B(hicma::laplacend, randx_B, transB?n:k, transB?k:n);
  const std::vector<std::vector<double>> randx_C{hicma::get_sorted_random_vector(m>n?m:n)};
  hicma::Dense C(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(C);
  hicma::gemm(A, B, C, alpha, beta, transA, transB);

  naive_gemm(A, B, C_check, transA, transB, alpha, beta);

  // Check result
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_DOUBLE_EQ(C_check(i, j), C(i, j));
    }
  }
}

TEST_P(GEMMTests, DenseGemmAssign) {
  int64_t m, n, k;
  bool transA, transB;
  double alpha, _;
  std::tie(m, k, n, transA, transB, alpha, _) = GetParam();

  hicma::initialize();
  const std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(m>k?m:k)};
  const hicma::Dense A(hicma::laplacend, randx_A, transA?k:m, transA?m:k);
  const std::vector<std::vector<double>> randx_B{hicma::get_sorted_random_vector(k>n?k:n)};
  const hicma::Dense B(hicma::laplacend, randx_B, transB?n:k, transB?k:n);
  const hicma::Dense C = hicma::gemm(A, B, alpha, transA, transB);

  // Manual matmul
  hicma::Dense C_check(C.dim[0], C.dim[1]);
  naive_gemm(A, B, C_check, transA, transB, alpha, 0);

  // Check result
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_DOUBLE_EQ(C_check(i, j), C(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Params, GEMMTests,
    testing::Combine(testing::Values(13), testing::Values(47),
                     testing::Values(55), testing::Bool(), testing::Bool(),
                     testing::Range(-1.0, 1.5, 0.5),
                     testing::Range(-1.0, 1.5, 0.5)),
    [](const testing::TestParamInfo<GEMMTests::ParamType>& info) {
      std::stringstream alpha_;
      std::stringstream beta_;
      alpha_ << std::fixed << std::setprecision(1) << std::get<5>(info.param);
      beta_ << std::fixed << std::setprecision(1) << std::get<6>(info.param);
      std::string alpha = alpha_.str();
      alpha.replace(alpha.find_last_of("."), 1, "l");
      if (alpha.find_last_of("-") < alpha.length())
        alpha.replace(alpha.find_last_of("-"), 1, "m");
      std::string beta = beta_.str();
      beta.replace(beta.find_last_of("."), 1, "l");
      if (beta.find_last_of("-") < beta.length())
        beta.replace(beta.find_last_of("-"), 1, "m");
      std::string name =
          ("TA" + std::to_string(std::get<3>(info.param)) + "TB" +
           std::to_string(std::get<4>(info.param)) + "A" + alpha + "B" + beta);
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    Sizes, GEMMTests,
    testing::Combine(testing::Values(16, 32, 64), testing::Values(16, 32, 64),
                     testing::Values(16, 32, 64), testing::Values(false),
                     testing::Values(false), testing::Values(1),
                     testing::Values(1)),
    [](const testing::TestParamInfo<GEMMTests::ParamType>& info) {
      std::string name = ("M" + std::to_string(std::get<0>(info.param)) + "K" +
                          std::to_string(std::get<1>(info.param)) + "N" +
                          std::to_string(std::get<2>(info.param)));
      return name;
    });
