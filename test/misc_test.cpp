#include <cstdint>
#include <vector>
#include <string>
#include <tuple>

#include "FRANK/FRANK.h"
#include "gtest/gtest.h"

class MakeZeroTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};

void EXPECT_DENSE_ZERO(const FRANK::Dense& A) {
  for(int64_t i = 0; i < A.dim[0]; i++) {
    for(int64_t j = 0; j < A.dim[1]; j++) {
      EXPECT_DOUBLE_EQ(A(i, j), 0.);
    }
  }
}

void EXPECT_LOWRANK_ZERO(const FRANK::LowRank& A) {
  EXPECT_DENSE_ZERO(A.S);
  for(int64_t i = 0; i < A.U.dim[0]; i++) {
    for(int64_t j = 0; j < A.U.dim[1]; j++) {
      if(i == j) EXPECT_DOUBLE_EQ(A.U(i, j), 1.);
      else EXPECT_DOUBLE_EQ(A.U(i, j), 0.);
    }
  }
  for(int64_t i = 0; i < A.V.dim[0]; i++) {
    for(int64_t j = 0; j < A.V.dim[1]; j++) {
      if(i == j) EXPECT_DOUBLE_EQ(A.V(i, j), 1.);
      else EXPECT_DOUBLE_EQ(A.V(i, j), 0.);
    }
  }
}

void EXPECT_HIERARCHICAL_ZERO(FRANK::Hierarchical& A) {
  for(int64_t i = 0; i < A.dim[0]; i++) {
    for(int64_t j = 0; j < A.dim[1]; j++) {
      if(FRANK::type(A(i, j)) == "Dense") {
        const FRANK::Dense Aij(std::move(A(i, j)));
        EXPECT_DENSE_ZERO(Aij);
      }
      else if(FRANK::type(A(i, j)) == "LowRank") {
        const FRANK::LowRank Aij(std::move(A(i, j)));
        EXPECT_LOWRANK_ZERO(Aij);
      }
      else {
        FRANK::Hierarchical Aij(std::move(A(i, j)));
        EXPECT_HIERARCHICAL_ZERO(Aij);
      }
    }
  }
}

TEST_P(MakeZeroTests, ZeroAll_Dense) {
  FRANK::initialize();
  int64_t n_rows, n_cols;
  std::tie(n_rows, n_cols) = GetParam();

  FRANK::Dense A(FRANK::random_normal, {}, n_rows, n_cols);
  FRANK::zero_all(A);
  EXPECT_DENSE_ZERO(A);
}

TEST_P(MakeZeroTests, ZeroAll_LowRank) {
  FRANK::initialize();
  int64_t n_rows, n_cols;
  std::tie(n_rows, n_cols) = GetParam();

  const std::vector<std::vector<double>> randx {
    FRANK::get_sorted_random_vector(4 * std::max(n_rows, n_cols))
  };
  const FRANK::Dense A_(FRANK::laplacend, randx, n_rows, n_cols, 0, 2 * n_cols);
  FRANK::LowRank A(A_, 1e-8);

  FRANK::zero_all(A);
  EXPECT_DENSE_ZERO(A.S);
  for(int64_t i = 0; i < A.U.dim[0]; i++) {
    for(int64_t j = 0; j < A.U.dim[1]; j++) {
      if(i == j) EXPECT_DOUBLE_EQ(A.U(i, j), 1.);
      else EXPECT_DOUBLE_EQ(A.U(i, j), 0.);
    }
  }
  for(int64_t i = 0; i < A.V.dim[0]; i++) {
    for(int64_t j = 0; j < A.V.dim[1]; j++) {
      if(i == j) EXPECT_DOUBLE_EQ(A.V(i, j), 1.);
      else EXPECT_DOUBLE_EQ(A.V(i, j), 0.);
    }
  }
}

TEST_P(MakeZeroTests, ZeroAll_Hierarchical) {
  FRANK::initialize();
  int64_t n_rows, n_cols;
  std::tie(n_rows, n_cols) = GetParam();

  const std::vector<std::vector<double>> randx {
    FRANK::get_sorted_random_vector(std::max(n_rows, n_cols))
  };
  constexpr int64_t nleaf = 8;
  constexpr int64_t nb_rows = 2;
  constexpr int64_t nb_cols = 2;
  constexpr double admis = 0;
  constexpr double eps = 1e-8;
  FRANK::Hierarchical A(FRANK::laplacend, randx, n_rows, n_cols,
                        nleaf, eps, admis, nb_rows, nb_cols, FRANK::AdmisType::PositionBased);

  FRANK::zero_all(A);
  EXPECT_HIERARCHICAL_ZERO(A);
}

void EXPECT_DENSE_UPPERTRI(const FRANK::Dense& A) {
  for(int64_t i = 0; i < A.dim[0]; i++) {
    for(int64_t j = 0; j < A.dim[1]; j++) {
      if(i <= j) EXPECT_NE(A(i, j), 0);
      else EXPECT_DOUBLE_EQ(A(i, j), 0);
    }
  }
}

void EXPECT_HIERARCHICAL_UPPERTRI(FRANK::Hierarchical& A) {
  for(int64_t i = 0; i < A.dim[0]; i++) {
    for(int64_t j = 0; j < A.dim[1]; j++) {
      if(i == j) {
        if(FRANK::type(A(i, j)) == "Dense") {
          const FRANK::Dense Aij(std::move(A(i, j)));
          EXPECT_DENSE_UPPERTRI(Aij);
        }
        else if(FRANK::type(A(i, j)) == "Hierarchical") {
          FRANK::Hierarchical Aij(std::move(A(i, j)));
          EXPECT_HIERARCHICAL_UPPERTRI(Aij);
        }
      }
      else if(i > j) {
        if(FRANK::type(A(i, j)) == "Dense") {
          const FRANK::Dense Aij(std::move(A(i, j)));
          EXPECT_DENSE_ZERO(Aij);
        }
        else if(FRANK::type(A(i, j)) == "LowRank") {
          const FRANK::LowRank Aij(std::move(A(i, j)));
          EXPECT_LOWRANK_ZERO(Aij);
        }
        else {
          FRANK::Hierarchical Aij(std::move(A(i, j)));
          EXPECT_HIERARCHICAL_ZERO(Aij);
        }
      }
    }
  }
}

TEST_P(MakeZeroTests, ZeroLower_Dense) {
  FRANK::initialize();
  int64_t n_rows, n_cols;
  std::tie(n_rows, n_cols) = GetParam();

  FRANK::Dense A(FRANK::random_normal, {}, n_rows, n_cols);
  FRANK::zero_lower(A);
  EXPECT_DENSE_UPPERTRI(A);
}

TEST_P(MakeZeroTests, ZeroLower_Hierarchical) {
  FRANK::initialize();
  int64_t n_rows, n_cols;
  std::tie(n_rows, n_cols) = GetParam();

  const std::vector<std::vector<double>> randx {
    FRANK::get_sorted_random_vector(std::max(n_rows, n_cols))
  };
  constexpr int64_t nleaf = 8;
  constexpr int64_t nb_rows = 2;
  constexpr int64_t nb_cols = 2;
  constexpr double admis = 0;
  constexpr double eps = 1e-8;
  FRANK::Hierarchical A(FRANK::laplacend, randx, n_rows, n_cols,
                        nleaf, eps, admis, nb_rows, nb_cols, FRANK::AdmisType::PositionBased);

  FRANK::zero_lower(A);
  EXPECT_HIERARCHICAL_UPPERTRI(A);
}

void EXPECT_DENSE_LOWERTRI(const FRANK::Dense& A) {
  for(int64_t i = 0; i < A.dim[0]; i++) {
    for(int64_t j = 0; j < A.dim[1]; j++) {
      if(i >= j) EXPECT_NE(A(i, j), 0);
      else EXPECT_DOUBLE_EQ(A(i, j), 0);
    }
  }
}

void EXPECT_HIERARCHICAL_LOWERTRI(FRANK::Hierarchical& A) {
  for(int64_t i = 0; i < A.dim[0]; i++) {
    for(int64_t j = 0; j < A.dim[1]; j++) {
      if(i == j) {
        if(FRANK::type(A(i, j)) == "Dense") {
          const FRANK::Dense Aij(std::move(A(i, j)));
          EXPECT_DENSE_LOWERTRI(Aij);
        }
        else if(FRANK::type(A(i, j)) == "Hierarchical") {
          FRANK::Hierarchical Aij(std::move(A(i, j)));
          EXPECT_HIERARCHICAL_LOWERTRI(Aij);
        }
      }
      else if(i < j) {
        if(FRANK::type(A(i, j)) == "Dense") {
          const FRANK::Dense Aij(std::move(A(i, j)));
          EXPECT_DENSE_ZERO(Aij);
        }
        else if(FRANK::type(A(i, j)) == "LowRank") {
          const FRANK::LowRank Aij(std::move(A(i, j)));
          EXPECT_LOWRANK_ZERO(Aij);
        }
        else {
          FRANK::Hierarchical Aij(std::move(A(i, j)));
          EXPECT_HIERARCHICAL_ZERO(Aij);
        }
      }
    }
  }
}

TEST_P(MakeZeroTests, ZeroUpper_Dense) {
  FRANK::initialize();
  int64_t n_rows, n_cols;
  std::tie(n_rows, n_cols) = GetParam();

  FRANK::Dense A(FRANK::random_normal, {}, n_rows, n_cols);
  FRANK::zero_upper(A);
  EXPECT_DENSE_LOWERTRI(A);
}

TEST_P(MakeZeroTests, ZeroUpper_Hierarchical) {
  FRANK::initialize();
  int64_t n_rows, n_cols;
  std::tie(n_rows, n_cols) = GetParam();

  const std::vector<std::vector<double>> randx {
    FRANK::get_sorted_random_vector(std::max(n_rows, n_cols))
  };
  constexpr int64_t nleaf = 8;
  constexpr int64_t nb_rows = 2;
  constexpr int64_t nb_cols = 2;
  constexpr double admis = 0;
  constexpr double eps = 1e-8;
  FRANK::Hierarchical A(FRANK::laplacend, randx, n_rows, n_cols,
                        nleaf, eps, admis, nb_rows, nb_cols, FRANK::AdmisType::PositionBased);

  FRANK::zero_upper(A);
  EXPECT_HIERARCHICAL_LOWERTRI(A);
}

INSTANTIATE_TEST_SUITE_P(MakeZeroTests, MakeZeroTests,
                         testing::Combine(testing::Values(64),
                                          testing::Values(32, 64, 128)
                                          ));
