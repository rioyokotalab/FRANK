#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include <iostream>

#include "hicma/hicma.h"
#include "gtest/gtest.h"


class HierarchicalTestFixedAccuracy
  : public testing::TestWithParam<std::tuple<int64_t, int64_t, double, double, int>> {};


TEST_P(HierarchicalTestFixedAccuracy, BLR_Construction_Kernel) {
  int64_t n_rows, n_cols, nleaf;
  double eps, admis;
  int admis_type;
  std::tie(n_rows, nleaf, eps, admis, admis_type) = GetParam();
  n_cols = n_rows;
  int64_t nb_row = n_rows/nleaf;
  int64_t nb_col = n_cols/nleaf;
  
  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(std::max(n_rows, n_cols))};

  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Hierarchical A(hicma::laplacend, randx_A, n_rows, n_cols,
			nleaf, eps, admis, nb_row, nb_col, admis_type);
  
  // Check compression error
  double error = hicma::l2_error(D, A);
  EXPECT_TRUE(error < eps);
}

TEST_P(HierarchicalTestFixedAccuracy, BLR_Construction_Block) {
  int64_t n_rows, n_cols, nleaf;
  double eps, admis;
  int admis_type;
  std::tie(n_rows, nleaf, eps, admis, admis_type) = GetParam();
  n_cols = n_rows;
  int64_t nb_row = n_rows/nleaf;
  int64_t nb_col = n_cols/nleaf;
  
  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(std::max(n_rows, n_cols))};
  
  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Dense D_copy(D);
  hicma::Hierarchical A(std::move(D_copy), nleaf, eps, admis,
			nb_row, nb_col, 0, 0, randx_A, admis_type);
  
  // Check compression error
  double error = hicma::l2_error(D, A);
  EXPECT_TRUE(error < eps);
}

TEST_P(HierarchicalTestFixedAccuracy, H_Construction_Kernel) {
  int64_t n_rows, n_cols, nleaf;
  double eps, admis;
  int admis_type;
  std::tie(n_rows, nleaf, eps, admis, admis_type) = GetParam();
  n_cols = n_rows;
  int64_t nb_row = 2;
  int64_t nb_col = 2;
  
  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(std::max(n_rows, n_cols))};

  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Hierarchical A(hicma::laplacend, randx_A, n_rows, n_cols,
			nleaf, eps, admis, nb_row, nb_col, admis_type);
  
  // Check compression error
  double error = hicma::l2_error(D, A);
  EXPECT_TRUE(error < eps);
}

TEST_P(HierarchicalTestFixedAccuracy, H_Construction_Block) {
  int64_t n_rows, n_cols, nleaf;
  double eps, admis;
  int admis_type;
  std::tie(n_rows, nleaf, eps, admis, admis_type) = GetParam();
  n_cols = n_rows;
  int64_t nb_row = 2;
  int64_t nb_col = 2;
  
  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(std::max(n_rows, n_cols))};
  
  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Dense D_copy(D);
  hicma::Hierarchical A(std::move(D_copy), nleaf, eps, admis,
			nb_row, nb_col, 0, 0, randx_A, admis_type);
  
  // Check compression error
  double error = hicma::l2_error(D, A);
  EXPECT_TRUE(error < eps);
}

INSTANTIATE_TEST_SUITE_P(Hierarchical, HierarchicalTestFixedAccuracy,
			 testing::Combine(testing::Values(128, 256),
					  testing::Values(16, 32),
					  testing::Values(1e-6, 1e-8, 1e-10),
					  testing::Values(0.0, 0.5, 1.0, 2.0),
					  testing::Values(POSITION_BASED_ADMIS, GEOMETRY_BASED_ADMIS)
					  ));

