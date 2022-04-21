#include <cstdint>
#include <vector>
#include <string>
#include <tuple>

#include "FRANK/FRANK.h"
#include "gtest/gtest.h"


class BLRFixedAccuracyTest
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, double, double, FRANK::AdmisType>> {
 protected:
  void SetUp() override {
    FRANK::initialize();
    FRANK::setGlobalValue("FRANK_LRA", "rounded_addition");
    std::tie(n_rows, nleaf, eps, admis, admis_type) = GetParam();
    n_cols = n_rows; // Assume square matrix
    nb_row = n_rows / nleaf;
    nb_col = n_cols / nleaf;
    randx_A.emplace_back(FRANK::get_sorted_random_vector(std::max(n_rows, n_cols)));
  }
  int64_t n_rows, n_cols, nb_row, nb_col, nleaf;
  double eps, admis;
  FRANK::AdmisType admis_type;
  std::vector<std::vector<double>> randx_A;
};

class BLRFixedAccuracyTest_AllowTallSkinny
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t, double, double, FRANK::AdmisType>> {
 protected:
  void SetUp() override {
    FRANK::initialize();
    FRANK::setGlobalValue("FRANK_LRA", "rounded_addition");
    std::tie(n_rows, n_cols, nleaf, eps, admis, admis_type) = GetParam();
    nb_row = n_rows / nleaf;
    nb_col = n_cols / nleaf;
    randx_A.emplace_back(FRANK::get_sorted_random_vector(std::max(n_rows, n_cols)));
  }
  int64_t n_rows, n_cols, nb_row, nb_col, nleaf;
  double eps, admis;
  FRANK::AdmisType admis_type;
  std::vector<std::vector<double>> randx_A;
};


TEST_P(BLRFixedAccuracyTest, ConstructionByKernel) {
  const FRANK::Dense D(FRANK::laplacend, randx_A, n_rows, n_cols);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, n_rows, n_cols,
                              nleaf, eps, admis, nb_row, nb_col, admis_type);

  // Check compression error
  const double error = FRANK::l2_error(D, A);
  EXPECT_LE(error, eps);
}

TEST_P(BLRFixedAccuracyTest, ConstructionByDenseMatrix) {
  const FRANK::Dense D(FRANK::laplacend, randx_A, n_rows, n_cols);
  FRANK::Dense D_copy(D);
  const FRANK::Hierarchical A(std::move(D_copy), nleaf, eps, admis,
                              nb_row, nb_col, 0, 0, randx_A, admis_type);

  // Check compression error
  const double error = FRANK::l2_error(D, A);
  EXPECT_LE(error, eps);
}

TEST_P(BLRFixedAccuracyTest, LUFactorization) {
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

TEST_P(BLRFixedAccuracyTest_AllowTallSkinny, GramSchmidtQRFactorization) {
  FRANK::Hierarchical A(FRANK::laplacend, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);
  const FRANK::Hierarchical D(FRANK::laplacend, randx_A, n_rows, n_cols,
                              nleaf, nleaf, nb_row, nb_row, nb_col, FRANK::AdmisType::PositionBased);

  FRANK::Hierarchical Q(A);
  FRANK::Hierarchical R(FRANK::zeros, randx_A, n_cols, n_cols,
                        nleaf, eps, admis, nb_col, nb_col, admis_type);
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

TEST_P(BLRFixedAccuracyTest_AllowTallSkinny, BlockedHouseholderQRFactorization) {
  FRANK::Hierarchical A(FRANK::laplacend, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);
  const FRANK::Hierarchical D(FRANK::laplacend, randx_A, n_rows, n_cols,
                              nleaf, nleaf, nb_row, nb_row, nb_col, FRANK::AdmisType::PositionBased);
  FRANK::Hierarchical T(A.dim[1], 1);
  FRANK::blocked_householder_blr_qr(A, T);

  FRANK::Hierarchical Q(FRANK::identity, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);
  FRANK::left_multiply_blocked_reflector(A, T, Q, false);

  // Residual
  FRANK::Hierarchical QR(Q);
  //R is taken from upper triangular part of A
  FRANK::Hierarchical R(nb_col, nb_col);
  for(int64_t i = 0; i < nb_col; i++) {
    for(int64_t j = i; j < nb_col; j++) {
      R(i, j) = A(i, j);
    }
  }
  FRANK::trmm(R, QR, FRANK::Side::Right, FRANK::Mode::Upper, 'n', 'n', 1.);
  const double residual = FRANK::l2_error(D, QR);
  EXPECT_LE(residual, eps);

  // Orthogonality
  FRANK::left_multiply_blocked_reflector(A, T, Q, true);
  // Take square part as Q^T x Q (assuming n_rows >= n_cols)
  FRANK::Hierarchical QtQ(nb_col, nb_col);
  for(int64_t i = 0; i < nb_col; i++) {
    for(int64_t j = 0; j < nb_col; j++) {
      QtQ(i, j) = Q(i, j);
    }
  }
  const double orthogonality = FRANK::l2_error(FRANK::Dense(FRANK::identity, {}, n_cols, n_cols), QtQ);
  EXPECT_LE(orthogonality, eps);
}

TEST_P(BLRFixedAccuracyTest_AllowTallSkinny, TiledHouseholderQRFactorization) {
  FRANK::Hierarchical A(FRANK::laplacend, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);
  const FRANK::Hierarchical D(FRANK::laplacend, randx_A, n_rows, n_cols,
                        nleaf, nleaf, nb_row, nb_row, nb_col, FRANK::AdmisType::PositionBased);
  FRANK::Hierarchical T(A.dim[0], A.dim[1]);
  for(int64_t j = 0; j < A.dim[1]; j++) {
    for(int64_t i = 0; i < A.dim[0]; i++) {
      T(i, j) = FRANK::Dense(i < j ? 0 : FRANK::get_n_cols(A(j, j)),
                             i < j ? 0 : FRANK::get_n_cols(A(j, j)));
    }
  }
  FRANK::tiled_householder_blr_qr(A, T);

  FRANK::Hierarchical Q(FRANK::identity, randx_A, n_rows, n_cols,
                        nleaf, eps, admis, nb_row, nb_col, admis_type);
  FRANK::left_multiply_tiled_reflector(A, T, Q, false);

  // Residual
  FRANK::Hierarchical QR(Q);
  //R is taken from upper triangular part of A
  FRANK::Hierarchical R(nb_col, nb_col);
  for(int64_t i = 0; i < nb_col; i++) {
    for(int64_t j = i; j < nb_col; j++) {
      R(i, j) = A(i, j);
    }
  }
  FRANK::trmm(R, QR, FRANK::Side::Right, FRANK::Mode::Upper, 'n', 'n', 1.);
  const double residual = FRANK::l2_error(D, QR);
  EXPECT_LE(residual, eps);

  // Orthogonality
  FRANK::left_multiply_tiled_reflector(A, T, Q, true);
  // Take square part as Q^T x Q (assuming n_rows >= n_cols)
  FRANK::Hierarchical QtQ(nb_col, nb_col);
  for(int64_t i = 0; i < nb_col; i++) {
    for(int64_t j = 0; j < nb_col; j++) {
      QtQ(i, j) = Q(i, j);
    }
  }
  const double orthogonality = FRANK::l2_error(FRANK::Dense(FRANK::identity, {}, n_cols, n_cols), QtQ);
  EXPECT_LE(orthogonality, eps);
}

INSTANTIATE_TEST_SUITE_P(BLRTest, BLRFixedAccuracyTest,
                         testing::Combine(testing::Values(128, 256),
                                          testing::Values(32),
                                          testing::Values(1e-6, 1e-8, 1e-10),
                                          testing::Values(0.0, 1.0, 4.0),
                                          testing::Values(FRANK::AdmisType::PositionBased, FRANK::AdmisType::GeometryBased)
                                          ));

INSTANTIATE_TEST_SUITE_P(BLRTest, BLRFixedAccuracyTest_AllowTallSkinny,
                         testing::Combine(testing::Values(256),
                                          testing::Values(128, 256),
                                          testing::Values(32),
                                          testing::Values(1e-6, 1e-8, 1e-10),
                                          testing::Values(0.0, 1.0, 4.0),
                                          testing::Values(FRANK::AdmisType::PositionBased, FRANK::AdmisType::GeometryBased)
                                          ));

