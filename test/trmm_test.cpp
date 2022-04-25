#include <cstdint>
#include <vector>
#include <string>
#include <tuple>

#include "FRANK/FRANK.h"
#include "gtest/gtest.h"

class TRMMTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, FRANK::Side, FRANK::Mode, char, char, double>> {
 protected:
  void SetUp() override {
    FRANK::initialize();
    FRANK::setGlobalValue("FRANK_LRA", "rounded_addition");
    std::tie(n_rows, n_cols, side, uplo, trans, diag, alpha) = GetParam();
  }
  int64_t n_rows, n_cols;
  FRANK::Side side;
  FRANK::Mode uplo;
  char trans, diag;
  double alpha;
};

void EXPECT_DENSE_NEAR(const FRANK::Dense& A, const FRANK::Dense& B, const double threshold) {
  EXPECT_EQ(A.dim[0], B.dim[0]);
  EXPECT_EQ(A.dim[1], B.dim[1]);
  for(int64_t i = 0; i < A.dim[0]; i++) {
    for(int64_t j = 0; j < A.dim[1]; j++) {
      EXPECT_NEAR(A(i, j), B(i, j), threshold);
    }
  }
}

void make_unit_diag(FRANK::Hierarchical& A) {
  for(int64_t k = 0; k < std::min(A.dim[0], A.dim[1]); k++) {
    if(FRANK::type(A(k, k)) == "Dense") {
      FRANK::Dense Akk(std::move(A(k, k)));
      for(int64_t i = 0; i < std::min(Akk.dim[0], Akk.dim[1]); i++) {
        Akk(i, i) = 1.0;
      }
      A(k, k) = std::move(Akk);
    }
    else if(FRANK::type(A(k, k)) == "Hierarchical") {
      FRANK::Hierarchical Akk(std::move(A(k, k)));
      make_unit_diag(Akk);
      A(k, k) = std::move(Akk);
    }
  }
}

TEST_P(TRMMTests, DenseDense) {
  FRANK::Dense A(FRANK::random_normal, {},
                 side == FRANK::Side::Left ? n_rows : n_cols,
                 side == FRANK::Side::Left ? n_rows : n_cols);
  FRANK::Dense B(FRANK::random_normal, {}, n_rows, n_cols);
  const FRANK::Dense B_copy(B);
  trmm(A, B, side, uplo, trans, diag, alpha);

  // Check result
  if(uplo == FRANK::Mode::Lower)
    FRANK::zero_upper(A);
  else
    FRANK::zero_lower(A);
  if(diag == 'u') {
    for(int64_t k = 0; k < std::min(A.dim[0], A.dim[1]); k++)
      A(k, k) = 1.0;
  }
  FRANK::Dense B_check(n_rows, n_cols);
  if(side == FRANK::Side::Left)
    FRANK::gemm(A, B_copy, B_check, alpha, 0, trans == 't', false);
  else
    FRANK::gemm(B_copy, A, B_check, alpha, 0, false, trans == 't');

  EXPECT_DENSE_NEAR(B, B_check, 1e-12);
}

TEST_P(TRMMTests, DenseLowrank) {
  FRANK::Dense A(FRANK::random_normal, {},
                 side == FRANK::Side::Left ? n_rows : n_cols,
                 side == FRANK::Side::Left ? n_rows : n_cols);
  const std::vector<std::vector<double>> randx {
    FRANK::get_sorted_random_vector(4 * std::max(n_rows, n_cols))
  };
  const FRANK::Dense B_(FRANK::laplacend, randx, n_rows, n_cols, 0, 2 * n_cols);
  FRANK::LowRank B(B_, 1e-8);
  const FRANK::LowRank B_copy(B);
  trmm(A, B, side, uplo, trans, diag, alpha);

  // Check result
  if(uplo == FRANK::Mode::Lower)
    FRANK::zero_upper(A);
  else
    FRANK::zero_lower(A);
  if(diag == 'u') {
    for(int64_t k = 0; k < std::min(A.dim[0], A.dim[1]); k++)
      A(k, k) = 1.0;
  }
  FRANK::LowRank B_check(B_copy);
  if(side == FRANK::Side::Left)
    FRANK::gemm(A, B_copy.U, B_check.U, alpha, 0, trans == 't', false);
  else
    FRANK::gemm(B_copy.V, A, B_check.V, alpha, 0, false, trans == 't');

  EXPECT_DENSE_NEAR(B.U, B_check.U, 1e-12);
  EXPECT_DENSE_NEAR(B.S, B_check.S, 1e-12);
  EXPECT_DENSE_NEAR(B.V, B_check.V, 1e-12);
}

TEST_P(TRMMTests, DenseHierarchical) {
  const std::vector<std::vector<double>> randx {
    FRANK::get_sorted_random_vector(std::max(n_rows, n_cols))
  };
  constexpr int64_t nleaf = 8;
  constexpr int64_t nb_rows = 2;
  constexpr int64_t nb_cols = 2;
  constexpr double admis = 0;
  constexpr double eps = 1e-10;
  FRANK::Dense A(FRANK::random_normal, {},
                 side == FRANK::Side::Left ? n_rows : n_cols,
                 side == FRANK::Side::Left ? n_rows : n_cols);
  FRANK::Hierarchical B(FRANK::laplacend, randx, n_rows, n_cols,
                        nleaf, eps, admis, nb_rows, nb_cols, FRANK::AdmisType::PositionBased);
  const FRANK::Hierarchical B_copy(B);
  trmm(A, B, side, uplo, trans, diag, alpha);

  // Check result
  if(uplo == FRANK::Mode::Lower)
    FRANK::zero_upper(A);
  else
    FRANK::zero_lower(A);
  if(diag == 'u') {
    for(int64_t k = 0; k < std::min(A.dim[0], A.dim[1]); k++)
      A(k, k) = 1.0;
  }
  FRANK::Hierarchical B_check(B_copy);
  if(side == FRANK::Side::Left)
    FRANK::gemm(A, B_copy, B_check, alpha, 0, trans == 't', false);
  else
    FRANK::gemm(B_copy, A, B_check, alpha, 0, false, trans == 't');

  const double diff = l2_error(B, B_check);
  EXPECT_LT(diff, eps);
}

TEST_P(TRMMTests, HierarchicalDense) {
  const std::vector<std::vector<double>> randx {
    FRANK::get_sorted_random_vector(std::max(n_rows, n_cols))
  };
  constexpr int64_t nleaf = 8;
  constexpr double admis = 0;
  constexpr double eps = 1e-10;
  const int64_t A_n = side == FRANK::Side::Left ? n_rows : n_cols;
  constexpr int64_t A_nb = 2;
  FRANK::Hierarchical A(FRANK::laplacend, randx, A_n, A_n,
                        nleaf, eps, admis, A_nb, A_nb, FRANK::AdmisType::PositionBased);
  FRANK::Dense B(FRANK::random_normal, {}, n_rows, n_cols);
  const FRANK::Dense B_copy(B);
  trmm(A, B, side, uplo, trans, diag, alpha);

  // Check result
  if(uplo == FRANK::Mode::Lower)
    FRANK::zero_upper(A);
  else
    FRANK::zero_lower(A);
  if(diag == 'u') {
    make_unit_diag(A);
  }
  FRANK::Dense B_check(B_copy);
  if(side == FRANK::Side::Left)
    FRANK::gemm(A, B_copy, B_check, alpha, 0, trans == 't', false);
  else
    FRANK::gemm(B_copy, A, B_check, alpha, 0, false, trans == 't');

  EXPECT_DENSE_NEAR(B, B_check, eps);
}

TEST_P(TRMMTests, HierarchicalLowrank) {
  const std::vector<std::vector<double>> randx {
    FRANK::get_sorted_random_vector(std::max(n_rows, n_cols))
  };
  constexpr int64_t nleaf = 8;
  constexpr double admis = 0;
  constexpr double eps = 1e-8;
  const int64_t A_n = side == FRANK::Side::Left ? n_rows : n_cols;
  constexpr int64_t A_nb = 2;
  FRANK::Hierarchical A(FRANK::laplacend, randx, A_n, A_n,
                        nleaf, eps, admis, A_nb, A_nb, FRANK::AdmisType::PositionBased);
  const FRANK::Dense B_(FRANK::laplacend, randx, n_rows, n_cols, 0, 2 * n_cols);
  FRANK::LowRank B(B_, eps);
  const FRANK::LowRank B_copy(B);
  trmm(A, B, side, uplo, trans, diag, alpha);

  // Check result
  if(uplo == FRANK::Mode::Lower)
    FRANK::zero_upper(A);
  else
    FRANK::zero_lower(A);
  if(diag == 'u') {
    make_unit_diag(A);
  }
  FRANK::LowRank B_check(B_copy);
  if(side == FRANK::Side::Left)
    FRANK::gemm(A, B_copy.U, B_check.U, alpha, 0, trans == 't', false);
  else
    FRANK::gemm(B_copy.V, A, B_check.V, alpha, 0, false, trans == 't');

  EXPECT_DENSE_NEAR(B.U, B_check.U, eps);
  EXPECT_DENSE_NEAR(B.S, B_check.S, eps);
  EXPECT_DENSE_NEAR(B.V, B_check.V, eps);
}

TEST_P(TRMMTests, HierarchicalHierarchical_Weak) {
  std::vector<std::vector<double>> randx {
    FRANK::get_sorted_random_vector(std::max(n_rows, n_cols))
  };
  constexpr int64_t nleaf = 8;
  constexpr int64_t nb_rows = 2;
  constexpr int64_t nb_cols = 2;
  constexpr double admis = 0;
  constexpr double eps = 1e-8;
  const int64_t A_n = side == FRANK::Side::Left ? n_rows : n_cols;
  constexpr int64_t A_nb = 2;
  FRANK::Hierarchical A(FRANK::laplacend, randx, A_n, A_n,
                        nleaf, eps, admis, A_nb, A_nb, FRANK::AdmisType::PositionBased);
  FRANK::Hierarchical B(FRANK::laplacend, randx, n_rows, n_cols,
                        nleaf, eps, admis, nb_rows, nb_cols, FRANK::AdmisType::PositionBased);
  const FRANK::Hierarchical B_copy(B);
  trmm(A, B, side, uplo, trans, diag, alpha);

  // Check result
  if(uplo == FRANK::Mode::Lower)
    FRANK::zero_upper(A);
  else
    FRANK::zero_lower(A);
  if(diag == 'u') {
    make_unit_diag(A);
  }
  FRANK::Hierarchical B_check(B_copy);
  if(side == FRANK::Side::Left)
    FRANK::gemm(A, B_copy, B_check, alpha, 0, trans == 't', false);
  else
    FRANK::gemm(B_copy, A, B_check, alpha, 0, false, trans == 't');

  const double diff = l2_error(B, B_check);
  EXPECT_LT(diff, eps);
}

TEST_P(TRMMTests, HierarchicalHierarchical_Strong) {
  std::vector<std::vector<double>> randx {
    FRANK::get_sorted_random_vector(std::max(n_rows, n_cols))
  };
  constexpr int64_t nleaf = 8;
  constexpr int64_t nb_rows = 2;
  constexpr int64_t nb_cols = 2;
  constexpr double admis = 1;
  constexpr double eps = 1e-8;
  const int64_t A_n = side == FRANK::Side::Left ? n_rows : n_cols;
  constexpr int64_t A_nb = 2;
  FRANK::Hierarchical A(FRANK::laplacend, randx, A_n, A_n,
                        nleaf, eps, admis, A_nb, A_nb, FRANK::AdmisType::PositionBased);
  FRANK::Hierarchical B(FRANK::laplacend, randx, n_rows, n_cols,
                        nleaf, eps, admis, nb_rows, nb_cols, FRANK::AdmisType::PositionBased);
  const FRANK::Hierarchical B_copy(B);
  trmm(A, B, side, uplo, trans, diag, alpha);

  // Check result
  if(uplo == FRANK::Mode::Lower)
    FRANK::zero_upper(A);
  else
    FRANK::zero_lower(A);
  if(diag == 'u') {
    make_unit_diag(A);
  }
  FRANK::Hierarchical B_check(B_copy);
  if(side == FRANK::Side::Left)
    FRANK::gemm(A, B_copy, B_check, alpha, 0, trans == 't', false);
  else
    FRANK::gemm(B_copy, A, B_check, alpha, 0, false, trans == 't');

  const double diff = l2_error(B, B_check);
  EXPECT_LT(diff, eps);
}

INSTANTIATE_TEST_SUITE_P(TRMMTests, TRMMTests,
                         testing::Combine(testing::Values(64),
                                          testing::Values(32, 64, 128),
                                          testing::Values(FRANK::Side::Left, FRANK::Side::Right),
                                          testing::Values(FRANK::Mode::Upper, FRANK::Mode::Lower),
                                          testing::Values('n'),
                                          testing::Values('u', 'n'),
                                          testing::Values(1.0, 2.0)
                                          ));
