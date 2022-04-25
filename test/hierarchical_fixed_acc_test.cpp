#include <cstdint>
#include <vector>
#include <string>
#include <tuple>

#include "FRANK/FRANK.h"
#include "gtest/gtest.h"


class HierarchicalFixedAccuracyTest
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, double, double, FRANK::AdmisType>> {
 protected:
  void SetUp() override {
    FRANK::initialize();
    FRANK::setGlobalValue("FRANK_LRA", "rounded_addition");
    std::tie(n_rows, nleaf, eps, admis, admis_type) = GetParam();
    n_cols = n_rows; // Assume square matrix
    nb_row = 2;
    nb_col = 2;
    randx_A.emplace_back(FRANK::get_sorted_random_vector(std::max(n_rows, n_cols)));
  }
  int64_t n_rows, n_cols, nb_row, nb_col, nleaf;
  double eps, admis;
  FRANK::AdmisType admis_type;
  std::vector<std::vector<double>> randx_A;
};


TEST_P(HierarchicalFixedAccuracyTest, ConstructionByKernel) {
  const FRANK::Dense D(FRANK::laplacend, randx_A, n_rows, n_cols);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, n_rows, n_cols,
                              nleaf, eps, admis, nb_row, nb_col, admis_type);

  // Check compression error
  const double error = FRANK::l2_error(D, A);
  EXPECT_LE(error, eps);
}

TEST_P(HierarchicalFixedAccuracyTest, ConstructionByDenseMatrix) {
  const FRANK::Dense D(FRANK::laplacend, randx_A, n_rows, n_cols);
  FRANK::Dense D_copy(D);
  const FRANK::Hierarchical A(std::move(D_copy), nleaf, eps, admis,
                              nb_row, nb_col, 0, 0, randx_A, admis_type);

  // Check compression error
  const double error = FRANK::l2_error(D, A);
  EXPECT_LE(error, eps);
}

TEST_P(HierarchicalFixedAccuracyTest, LUFactorization) {
  FRANK::Hierarchical A(FRANK::laplacend, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);

  const FRANK::Dense x(FRANK::random_uniform, {}, n_cols, 1);
  FRANK::Dense b(n_rows);
  FRANK::gemm(A, x, b, 1, 1);

  FRANK::Hierarchical L, U;
  std::tie(L, U) = FRANK::getrf(A);
  FRANK::trsm(L, b, FRANK::Mode::Lower);
  FRANK::trsm(U, b, FRANK::Mode::Upper);
  const double solve_error = FRANK::l2_error(x, b);

  // Check result
  EXPECT_LE(solve_error, eps);
}

TEST_P(HierarchicalFixedAccuracyTest, GramSchmidtQRFactorization) {
  FRANK::Hierarchical A(FRANK::laplacend, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);
  const FRANK::Hierarchical D(FRANK::laplacend, randx_A, n_rows, n_cols,
                              nleaf, nleaf, nb_row, nb_row, nb_col, FRANK::AdmisType::PositionBased);

  FRANK::Hierarchical Q(A);
  FRANK::Hierarchical R(A);
  FRANK::mgs_qr(A, Q, R);
  // Residual
  FRANK::Hierarchical QR(Q);
  FRANK::trmm(R, QR, FRANK::Side::Right, FRANK::Mode::Upper, 'n', 'n', 1.);
  const double residual = FRANK::l2_error(D, QR);
  EXPECT_LE(residual, eps);

  // Orthogonality
  FRANK::Hierarchical QtQ(FRANK::zeros, randx_A, n_cols, n_cols,
                          nleaf, eps, admis, nb_col, nb_col, admis_type);
  const FRANK::Hierarchical Qt = FRANK::transpose(Q);
  FRANK::gemm(Qt, Q, QtQ, 1, 0);
  const double orthogonality = FRANK::l2_error(FRANK::Dense(FRANK::identity, randx_A, n_cols, n_cols), QtQ);
  EXPECT_LE(orthogonality, eps);
}


INSTANTIATE_TEST_SUITE_P(HierarchicalTest, HierarchicalFixedAccuracyTest,
                         testing::Combine(testing::Values(128, 256),
                                          testing::Values(32),
                                          testing::Values(1e-6, 1e-8, 1e-10),
                                          testing::Values(0.0, 1.0, 4.0),
                                          testing::Values(FRANK::AdmisType::PositionBased, FRANK::AdmisType::GeometryBased)
                                          ));

