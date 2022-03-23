#include <cstdint>
#include <vector>
#include <string>
#include <tuple>

#include "hicma/hicma.h"
#include "gtest/gtest.h"


class BLRFixedAccuracyTest
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, double, double, int>> {
 protected:
  void SetUp() override {
    hicma::initialize();
    hicma::setGlobalValue("HICMA_LRA", "rounded_orth");
    std::tie(n_rows, nleaf, eps, admis, admis_type) = GetParam();
    n_cols = n_rows; // Assume square matrix
    nb_row = n_rows / nleaf;
    nb_col = n_cols / nleaf;
    randx_A.emplace_back(hicma::get_sorted_random_vector(std::max(n_rows, n_cols)));
  }
  int64_t n_rows, n_cols, nb_row, nb_col, nleaf;
  double eps, admis;
  int admis_type;
  std::vector<std::vector<double>> randx_A;
};


TEST_P(BLRFixedAccuracyTest, ConstructionByKernel) {
  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Hierarchical A(hicma::laplacend, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);

  // Check compression error
  double error = hicma::l2_error(D, A);
  EXPECT_LE(error, eps);
}

TEST_P(BLRFixedAccuracyTest, ConstructionByDenseMatrix) {
  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Dense D_copy(D);
  hicma::Hierarchical A(std::move(D_copy), nleaf, eps, admis,
                        nb_row, nb_col, 0, 0, randx_A, admis_type);

  // Check compression error
  double error = hicma::l2_error(D, A);
  EXPECT_LE(error, eps);
}

TEST_P(BLRFixedAccuracyTest, LUFactorization) {
  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Hierarchical A(hicma::laplacend, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);

  hicma::Dense x(hicma::random_uniform, std::vector<std::vector<double>>(), n_cols, 1);
  hicma::Dense b(n_rows);
  hicma::gemm(A, x, b, 1, 1);

  hicma::Hierarchical L, U;
  std::tie(L, U) = hicma::getrf(A);
  hicma::trsm(L, b, hicma::TRSM_LOWER);
  hicma::trsm(U, b, hicma::TRSM_UPPER);
  double solve_error = hicma::l2_error(x, b);

  // Check result
  EXPECT_LE(solve_error, eps);
}

INSTANTIATE_TEST_SUITE_P(BLRTest, BLRFixedAccuracyTest,
                         testing::Combine(testing::Values(128, 256),
                                          testing::Values(16, 32),
                                          testing::Values(1e-6, 1e-8, 1e-10),
                                          testing::Values(0.0, 1.0, 2.0),
                                          testing::Values(POSITION_BASED_ADMIS, GEOMETRY_BASED_ADMIS)
                                          ));

