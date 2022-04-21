#include "FRANK/FRANK.h"

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
      FRANK::initialize();
      const std::vector<std::vector<double>> randx_A{FRANK::get_sorted_random_vector(2*n)};
      A = FRANK::Dense(FRANK::laplacend, randx_A, n, n, 0, n);
      A_work = A;
    }
    FRANK::Dense A, A_work, A_check;
};

TEST_P(IDTests, ID) {
  int64_t n, rank;
  std::tie(n, rank) = GetParam();

  FRANK::Dense U, S, V;
  std::tie(U, S, V) = FRANK::id(A_work, rank);
  A_check = FRANK::gemm(FRANK::gemm(U, S), V);

  const double error = l2_error(A, A_check);
  EXPECT_LT(error, 1e-7);
}

TEST_P(IDTests, OID) {
  int64_t n, rank;
  std::tie(n, rank) = GetParam();

  std::vector<int64_t> indices;
  FRANK::Dense V;
  std::tie(V, indices) = FRANK::one_sided_id(A_work, rank);
  const FRANK::Dense A_cols = get_cols(A, indices);

  A_check = gemm(A_cols, V);
  const double error = l2_error(A, A_check);
  EXPECT_LT(error, 1e-7);
}

TEST_P(IDTests, RID) {
  int64_t n, rank;
  std::tie(n, rank) = GetParam();

  FRANK::Dense U, S, V;
  std::tie(U, S, V) = FRANK::rid(A_work, rank+5, rank);
  A_check = FRANK::gemm(FRANK::gemm(U, S), V);

  const double error = l2_error(A, A_check);
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
