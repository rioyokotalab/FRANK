#include <cstdint>
#include <vector>
#include <string>
#include <tuple>

#include "hicma/hicma.h"
#include "gtest/gtest.h"


class HierarchicalFixedAccuracyTest
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, double, double, hicma::AdmisType>> {
 protected:
  void SetUp() override {
    hicma::initialize();
    hicma::setGlobalValue("HICMA_LRA", "rounded_addition");
    std::tie(n_rows, nleaf, eps, admis, admis_type) = GetParam();
    n_cols = n_rows; // Assume square matrix
    nb_row = 2;
    nb_col = 2;
    randx_A.emplace_back(hicma::get_sorted_random_vector(std::max(n_rows, n_cols)));
  }
  int64_t n_rows, n_cols, nb_row, nb_col, nleaf;
  double eps, admis;
  hicma::AdmisType admis_type;
  std::vector<std::vector<double>> randx_A;
};


TEST_P(HierarchicalFixedAccuracyTest, ConstructionByKernel) {
  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Hierarchical A(hicma::laplacend, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);

  // Check compression error
  double error = hicma::l2_error(D, A);
  EXPECT_LE(error, eps);
}

TEST_P(HierarchicalFixedAccuracyTest, ConstructionByDenseMatrix) {
  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Dense D_copy(D);
  hicma::Hierarchical A(std::move(D_copy), nleaf, eps, admis,
                        nb_row, nb_col, 0, 0, randx_A, admis_type);

  // Check compression error
  double error = hicma::l2_error(D, A);
  EXPECT_LE(error, eps);
}

TEST_P(HierarchicalFixedAccuracyTest, LUFactorization) {
  hicma::Dense D(hicma::laplacend, randx_A, n_rows, n_cols);
  hicma::Hierarchical A(hicma::laplacend, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);

  hicma::Dense x(hicma::random_uniform, {}, n_cols, 1);
  hicma::Dense b(n_rows);
  hicma::gemm(A, x, b, 1, 1);

  hicma::Hierarchical L, U;
  std::tie(L, U) = hicma::getrf(A);
  hicma::trsm(L, b, hicma::Mode::Lower);
  hicma::trsm(U, b, hicma::Mode::Upper);
  double solve_error = hicma::l2_error(x, b);

  // Check result
  EXPECT_LE(solve_error, eps);
}

TEST_P(HierarchicalFixedAccuracyTest, GramSchmidtQRFactorization) {
  hicma::Hierarchical A(hicma::laplacend, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);
  hicma::Hierarchical D(hicma::laplacend, randx_A, n_rows, n_cols,
                        nleaf, nleaf, nb_row, nb_row, nb_col, hicma::AdmisType::PositionBased);

  hicma::Hierarchical Q(A);
  hicma::Hierarchical R(A);
  hicma::mgs_qr(A, Q, R);
  // Residual
  hicma::Hierarchical QR(Q);
  hicma::trmm(R, QR, hicma::Side::Right, hicma::Mode::Upper, 'n', 'n', 1.);
  double residual = hicma::l2_error(D, QR);
  EXPECT_LE(residual, eps);

  // Orthogonality
  hicma::Hierarchical QtQ(hicma::zeros, randx_A, n_cols, n_cols,
                          nleaf, eps, admis, nb_col, nb_col, admis_type);
  hicma::Hierarchical Qt = hicma::transpose(Q);
  hicma::gemm(Qt, Q, QtQ, 1, 0);
  double orthogonality = hicma::l2_error(hicma::Dense(hicma::identity, randx_A, n_cols, n_cols), QtQ);
  EXPECT_LE(orthogonality, eps);
}


INSTANTIATE_TEST_SUITE_P(HierarchicalTest, HierarchicalFixedAccuracyTest,
                         testing::Combine(testing::Values(128, 256),
                                          testing::Values(32),
                                          testing::Values(1e-6, 1e-8, 1e-10),
                                          testing::Values(0.0, 1.0, 4.0),
                                          testing::Values(hicma::AdmisType::PositionBased, hicma::AdmisType::GeometryBased)
                                          ));

