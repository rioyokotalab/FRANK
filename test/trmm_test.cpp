#include <cstdint>
#include <vector>
#include <string>
#include <tuple>

#include "hicma/hicma.h"
#include "gtest/gtest.h"

class TRMMTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, hicma::Side, hicma::Mode, char, char, double>> {
 protected:
  void SetUp() override {
    hicma::initialize();
    hicma::setGlobalValue("HICMA_LRA", "rounded_addition");
    std::tie(n_rows, n_cols, side, uplo, trans, diag, alpha) = GetParam();
  }
  int64_t n_rows, n_cols;
  hicma::Side side;
  hicma::Mode uplo;
  char trans, diag;
  double alpha;
};

void EXPECT_DENSE_NEAR(const hicma::Dense& A, const hicma::Dense& B, double threshold) {
  EXPECT_EQ(A.dim[0], B.dim[0]);
  EXPECT_EQ(A.dim[1], B.dim[1]);
  for(int64_t i = 0; i < A.dim[0]; i++) {
    for(int64_t j = 0; j < A.dim[1]; j++) {
      EXPECT_NEAR(A(i, j), B(i, j), threshold);
    }
  }
}

void make_unit_diag(hicma::Hierarchical& A) {
  for(int64_t k = 0; k < std::min(A.dim[0], A.dim[1]); k++) {
    if(hicma::type(A(k, k)) == "Dense") {
      hicma::Dense Akk(std::move(A(k, k)));
      for(int64_t i = 0; i < std::min(Akk.dim[0], Akk.dim[1]); i++) {
        Akk(i, i) = 1.0;
      }
      A(k, k) = std::move(Akk);
    }
    else if(hicma::type(A(k, k)) == "Hierarchical") {
      hicma::Hierarchical Akk(std::move(A(k, k)));
      make_unit_diag(Akk);
      A(k, k) = std::move(Akk);
    }
  }
}

TEST_P(TRMMTests, DenseDense) {
  hicma::Dense A(hicma::random_normal, {},
                 side == hicma::Left ? n_rows : n_cols,
                 side == hicma::Left ? n_rows : n_cols);
  hicma::Dense B(hicma::random_normal, {}, n_rows, n_cols);
  hicma::Dense B_copy(B);
  trmm(A, B, side, uplo, trans, diag, alpha);

  // Check result
  if(uplo == hicma::Lower)
    hicma::zero_upper(A);
  else
    hicma::zero_lower(A);
  if(diag == 'u') {
    for(int64_t k = 0; k < std::min(A.dim[0], A.dim[1]); k++)
      A(k, k) = 1.0;
  }
  hicma::Dense B_check(n_rows, n_cols);
  if(side == hicma::Left)
    hicma::gemm(A, B_copy, B_check, alpha, 0, trans == 't', false);
  else
    hicma::gemm(B_copy, A, B_check, alpha, 0, false, trans == 't');

  EXPECT_DENSE_NEAR(B, B_check, 1e-12);
}

TEST_P(TRMMTests, DenseLowrank) {
  hicma::Dense A(hicma::random_normal, {},
                 side == hicma::Left ? n_rows : n_cols,
                 side == hicma::Left ? n_rows : n_cols);
  std::vector<std::vector<double>> randx {
    hicma::get_sorted_random_vector(4 * std::max(n_rows, n_cols))
  };
  hicma::Dense B_(hicma::laplacend, randx, n_rows, n_cols, 0, 2 * n_cols);
  hicma::LowRank B(B_, 1e-8);
  hicma::LowRank B_copy(B);
  trmm(A, B, side, uplo, trans, diag, alpha);

  // Check result
  if(uplo == hicma::Lower)
    hicma::zero_upper(A);
  else
    hicma::zero_lower(A);
  if(diag == 'u') {
    for(int64_t k = 0; k < std::min(A.dim[0], A.dim[1]); k++)
      A(k, k) = 1.0;
  }
  hicma::LowRank B_check(B_copy);
  if(side == hicma::Left)
    hicma::gemm(A, B_copy.U, B_check.U, alpha, 0, trans == 't', false);
  else
    hicma::gemm(B_copy.V, A, B_check.V, alpha, 0, false, trans == 't');

  EXPECT_DENSE_NEAR(B.U, B_check.U, 1e-12);
  EXPECT_DENSE_NEAR(B.S, B_check.S, 1e-12);
  EXPECT_DENSE_NEAR(B.V, B_check.V, 1e-12);
}

TEST_P(TRMMTests, HierarchicalHierarchical_BLR) {
  std::vector<std::vector<double>> randx {
    hicma::get_sorted_random_vector(std::max(n_rows, n_cols))
  };
  int64_t nleaf = 16;
  int64_t nb_rows = n_rows / nleaf;
  int64_t nb_cols = n_cols / nleaf;
  double admis = 0;
  double eps = 1e-10;
  int64_t A_n = side == hicma::Left ? n_rows : n_cols;
  int64_t A_nb = A_n / nleaf;
  hicma::Hierarchical A(hicma::laplacend, randx, A_n, A_n,
                        nleaf, eps, admis, A_nb, A_nb, hicma::PositionBasedAdmis);
  hicma::Hierarchical B(hicma::laplacend, randx, n_rows, n_cols,
                        nleaf, eps, admis, nb_rows, nb_cols, hicma::PositionBasedAdmis);
  hicma::Hierarchical B_copy(B);
  trmm(A, B, side, uplo, trans, diag, alpha);

  // Check result
  if(uplo == hicma::Lower)
    hicma::zero_upper(A);
  else
    hicma::zero_lower(A);
  if(diag == 'u') {
    make_unit_diag(A);
  }
  hicma::Hierarchical B_check(B_copy);
  zero_all(B_check);
  if(side == hicma::Left)
    hicma::gemm(A, B_copy, B_check, alpha, 1, trans == 't', false);
  else
    hicma::gemm(B_copy, A, B_check, alpha, 1, false, trans == 't');

  double diff = l2_error(B, B_check);
  EXPECT_LT(diff, eps);
}

INSTANTIATE_TEST_SUITE_P(TRMMTests, TRMMTests,
                         testing::Combine(testing::Values(64),
                                          testing::Values(32, 64, 128),
                                          testing::Values(hicma::Left, hicma::Right),
                                          testing::Values(hicma::Upper, hicma::Lower),
                                          testing::Values('n'),
                                          testing::Values('u', 'n'),
                                          testing::Values(1.0, 2.0)
                                          ));
