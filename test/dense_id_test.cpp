#include "hicma/hicma.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>


class IDTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {
    protected:
    void SetUp() override {
      int64_t n, rank;
      std::tie(n, rank) = GetParam();
      hicma::initialize();
      std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(2*n)};
      A = hicma::Dense(hicma::laplacend, randx_A, n, n, 0, n);
      A_work = A;
    }
    hicma::Dense A, A_work, A_check;
};

TEST_P(IDTests, ID) {
  int64_t n, rank;
  std::tie(n, rank) = GetParam();

  hicma::Dense U, S, V;
  std::tie(U, S, V) = hicma::id(A_work, rank);
  A_check = hicma::gemm(hicma::gemm(U, S), V);

  double error = l2_error(A, A_check);
  EXPECT_LT(error, 1e-7);
}

TEST_P(IDTests, OID) {
  int64_t n, rank;
  std::tie(n, rank) = GetParam();

  std::vector<int64_t> indices;
  hicma::Dense V;
  std::tie(V, indices) = hicma::one_sided_id(A_work, rank);
  hicma::Dense A_cols = get_cols(A, indices);

  A_check = gemm(A_cols, V);
  double error = l2_error(A, A_check);
  EXPECT_LT(error, 1e-7);
}

TEST_P(IDTests, RID) {
  int64_t n, rank;
  std::tie(n, rank) = GetParam();

  hicma::Dense U, S, V;
  std::tie(U, S, V) = hicma::rid(A_work, rank+5, rank);
  A_check = hicma::gemm(hicma::gemm(U, S), V);

  double error = l2_error(A, A_check);
  EXPECT_LT(error, 1e-7);
}

INSTANTIATE_TEST_SUITE_P(
    Low_Rank, IDTests,
    testing::Values(std::make_tuple(2048, 16)),    
    [](const testing::TestParamInfo<IDTests::ParamType>& info) {
      std::string name = ("n" + std::to_string(std::get<0>(info.param)) + "rank" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });
