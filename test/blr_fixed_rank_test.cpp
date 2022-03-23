#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include <numeric>

#include "hicma/hicma.h"
#include "gtest/gtest.h"


class BLRFixedRankTest
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t, double, int>> {
 protected:
  void SetUp() override {
    hicma::initialize();
    std::tie(n_rows, nleaf, rank, admis, admis_type) = GetParam();
    n_cols = n_rows; // Assume square matrix
    nb_row = n_rows / nleaf;
    nb_col = n_cols / nleaf;
    randx_A.emplace_back(hicma::get_sorted_random_vector(std::max(n_rows, n_cols)));
  }
  int64_t n_rows, n_cols, nb_row, nb_col, nleaf, rank;
  double admis;
  int admis_type;
  std::vector<std::vector<double>> randx_A;
};

void expect_uniform_rank(hicma::Hierarchical& A, int64_t rank) {
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<A.dim[1]; j++) {
      if(hicma::type(A(i, j)) == "LowRank") {
        hicma::LowRank Aij(std::move(A(i, j)));
        EXPECT_EQ(Aij.rank, rank);
      }
      else if(hicma::type(A(i, j)) == "Hierarchical") {
        hicma::Hierarchical Aij(std::move(A(i, j)));
        expect_uniform_rank(Aij, rank);
      }
    }
  }
}


TEST_P(BLRFixedRankTest, ConstructionByKernel) {
  hicma::Hierarchical A(hicma::laplacend, randx_A, n_rows, n_cols,
                        rank, nleaf, admis, nb_row, nb_col, admis_type);
  expect_uniform_rank(A, rank);
}

TEST_P(BLRFixedRankTest, ConstructionByDenseMatrix) {
  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Hierarchical A(std::move(D), rank, nleaf, admis,
                        nb_row, nb_col, 0, 0, randx_A, admis_type);
  expect_uniform_rank(A, rank);
}

TEST_P(BLRFixedRankTest, LUFactorization) {
  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Hierarchical A(hicma::laplacend, randx_A, n_rows, n_cols,
                        rank, nleaf, admis, nb_row, nb_col, admis_type);
  double compress_error = hicma::l2_error(A, D);

  hicma::Dense x(hicma::random_uniform, std::vector<std::vector<double>>(), n_cols, 1);
  hicma::Dense b(n_rows);
  hicma::gemm(A, x, b, 1, 1);

  hicma::Hierarchical L, U;
  std::tie(L, U) = hicma::getrf(A);
  hicma::trsm(L, b, hicma::TRSM_LOWER);
  hicma::trsm(U, b, hicma::TRSM_UPPER);
  double solve_error = hicma::l2_error(x, b);

  // Check result
  double err_ratio = solve_error / (compress_error + std::numeric_limits<double>::epsilon());
  EXPECT_LE(err_ratio, 1e+2);
}

INSTANTIATE_TEST_SUITE_P(BLRTest, BLRFixedRankTest,
                         testing::Combine(testing::Values(128, 256),
                                          testing::Values(16, 32),
                                          testing::Values(4, 8),
                                          testing::Values(0.0, 0.5, 1.0, 2.0),
                                          testing::Values(POSITION_BASED_ADMIS, GEOMETRY_BASED_ADMIS)
                                          ));

