#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include <limits>

#include "FRANK/FRANK.h"
#include "gtest/gtest.h"

class GEMMTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t, double, double, bool, bool>> {
 protected:
  void SetUp() override {
    FRANK::initialize();
    FRANK::setGlobalValue("FRANK_LRA", "rounded_addition");
    std::tie(m, k, n, alpha, beta, transA, transB) = GetParam();
    randx_A = { FRANK::get_sorted_random_vector(4 * std::max(m, k)) };
    randx_B = { FRANK::get_sorted_random_vector(4 * std::max(k, n)) };
    randx_C = { FRANK::get_sorted_random_vector(4 * std::max(m, n)) };
  }
  int64_t m, k, n;
  double alpha, beta;
  bool transA, transB;
  std::vector<std::vector<double>> randx_A, randx_B, randx_C;
};

constexpr double THRESHOLD = 1e-6;
constexpr double EPSILON = 1e-14;
constexpr int64_t nleaf = 4;
constexpr int64_t nblocks = 2;
constexpr double admis = 0;

static void naive_gemm(const FRANK::Dense &A, const FRANK::Dense &B, FRANK::Dense &C,
                       const double alpha, const double beta, const bool transA, const bool transB) {
  for (int64_t i = 0; i < C.dim[0]; i++) {
    for (int64_t j = 0; j < C.dim[1]; j++) {
      C(i, j) =
          (beta * C(i, j) +
           alpha * (transA ? A(0, i) : A(i, 0)) * (transB ? B(j, 0) : B(0, j)));
      for (int64_t k = 1; k < (transA?A.dim[0]:A.dim[1]); k++) {
        C(i, j) +=
            (alpha * (transA ? A(k, i) : A(i, k)) * (transB ? B(j, k) : B(k, j)));
      }
    }
  }
}

TEST_P(GEMMTests, DenseDenseDense) {
  //D D D
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  FRANK::Dense C(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(C);
  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  
  naive_gemm(A, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, EPSILON);
}

TEST_P(GEMMTests, DenseDenseLowrank) {
  //D D LR
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  FRANK::Dense C_check(CD);
  FRANK::LowRank C(CD, THRESHOLD);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseDenseHierarchical) {
  //D D H
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(CD);
  FRANK::Hierarchical C(FRANK::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseLowrankDense) {
  //D LR D
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);
  
  FRANK::Dense C(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(C);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseLowrankLowrank) {
  //D LR LR
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  FRANK::Dense C_check(CD);
  FRANK::LowRank C(CD, THRESHOLD);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseLowrankHierarchical) {
  //D LR LR
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(CD);
  FRANK::Hierarchical C(FRANK::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseHierarchicalDense) {
  //D H D
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense C(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(C);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseHierarchicalLowrank) {
  //D H LR
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  FRANK::Dense C_check(CD);
  FRANK::LowRank C(CD, THRESHOLD);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseHierarchicalHierarchical) {
  //D H H
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(CD);
  FRANK::Hierarchical C(FRANK::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankDenseDense) {
  //LR D D
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);

  FRANK::Dense C(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(C);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankDenseLowrank) {
  //LR D LR
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  FRANK::Dense C_check(CD);
  FRANK::LowRank C(CD, THRESHOLD);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankDenseHierarchical) {
  //LR D H
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(CD);
  FRANK::Hierarchical C(FRANK::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankLowrankDense) {
  //LR LR D
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense C(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(C);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankLowrankLowrank) {
  //LR LR LR
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  FRANK::Dense C_check(CD);
  FRANK::LowRank C(CD, THRESHOLD);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankLowrankHierarchical) {
  //LR LR H
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(CD);
  FRANK::Hierarchical C(FRANK::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankHierarchicalDense) {
  //LR H D
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense C(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(C);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankHierarchicalLowrank) {
  //LR H LR
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  FRANK::Dense C_check(CD);
  FRANK::LowRank C(CD, THRESHOLD);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankHierarchicalHierarchical) {
  //LR H H
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(CD);
  FRANK::Hierarchical C(FRANK::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalDenseDense) {
  //H D D
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);

  FRANK::Dense C(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(C);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalDenseLowrank) {
  //H D LR
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  FRANK::Dense C_check(CD);
  FRANK::LowRank C(CD, THRESHOLD);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalDenseHierarchical) {
  //H D H
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(CD);
  FRANK::Hierarchical C(FRANK::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalLowrankDense) {
  //H LR D
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense C(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(C);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalLowrankLowrank) {
  //H LR LR
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  FRANK::Dense C_check(CD);
  FRANK::LowRank C(CD, THRESHOLD);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalLowrankHierarchical) {
  //H LR H
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(CD);
  FRANK::Hierarchical C(FRANK::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalHierarchicalDense) {
  //H H D
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense C(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(C);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalHierarchicalLowrank) {
  //H H LR
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  FRANK::Dense C_check(CD);
  FRANK::LowRank C(CD, THRESHOLD);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalHierarchicalHierarchical) {
  //H H H
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense CD(FRANK::laplacend, randx_C, m, n);
  FRANK::Dense C_check(CD);
  FRANK::Hierarchical C(FRANK::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseDenseNewDense) {
  //D D New(D)
  if(beta != 0) GTEST_SKIP();
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);

  FRANK::Dense C_check(m, n);
  FRANK::Dense C = gemm(A, B, alpha, transA, transB);
  naive_gemm(A, B, C_check, alpha, 0, transA, transB);

  // Check result
  EXPECT_EQ(C.dim[0], C_check.dim[0]);
  EXPECT_EQ(C.dim[1], C_check.dim[1]);
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, EPSILON);
}

TEST_P(GEMMTests, DenseLowrankNewDense) {
  //D LR New(D)
  if(beta != 0) GTEST_SKIP();
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense C_check(m, n);
  FRANK::Dense C = gemm(A, B, alpha, transA, transB);
  naive_gemm(A, BD, C_check, alpha, 0, transA, transB);

  // Check result
  EXPECT_EQ(C.dim[0], C_check.dim[0]);
  EXPECT_EQ(C.dim[1], C_check.dim[1]);
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseHierarchicalNewDense) {
  //D H New(D)
  if(beta != 0) GTEST_SKIP();
  const FRANK::Dense A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense C_check(m, n);
  FRANK::Dense C = gemm(A, B, alpha, transA, transB);
  naive_gemm(A, BD, C_check, alpha, 0, transA, transB);

  // Check result
  EXPECT_EQ(C.dim[0], C_check.dim[0]);
  EXPECT_EQ(C.dim[1], C_check.dim[1]);
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankDenseNewDense) {
  //LR D New(D)
  if(beta != 0) GTEST_SKIP();
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);

  FRANK::Dense C_check(m, n);
  FRANK::Dense C = gemm(A, B, alpha, transA, transB);
  naive_gemm(AD, B, C_check, alpha, 0, transA, transB);

  // Check result
  EXPECT_EQ(C.dim[0], C_check.dim[0]);
  EXPECT_EQ(C.dim[1], C_check.dim[1]);
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankLowrankNewDense) {
  //LR LR New(D)
  if(beta != 0) GTEST_SKIP();
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense C_check(m, n);
  FRANK::Dense C = gemm(A, B, alpha, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, 0, transA, transB);

  // Check result
  EXPECT_EQ(C.dim[0], C_check.dim[0]);
  EXPECT_EQ(C.dim[1], C_check.dim[1]);
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankHierarchicalNewDense) {
  //LR H New(D)
  if(beta != 0) GTEST_SKIP();
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const FRANK::LowRank A(AD, THRESHOLD);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense C_check(m, n);
  FRANK::Dense C = gemm(A, B, alpha, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, 0, transA, transB);

  // Check result
  EXPECT_EQ(C.dim[0], C_check.dim[0]);
  EXPECT_EQ(C.dim[1], C_check.dim[1]);
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalDenseNewDense) {
  //H D New(D)
  if(beta != 0) GTEST_SKIP();
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);

  FRANK::Dense C_check(m, n);
  FRANK::Dense C = gemm(A, B, alpha, transA, transB);
  naive_gemm(AD, B, C_check, alpha, 0, transA, transB);

  // Check result
  EXPECT_EQ(C.dim[0], C_check.dim[0]);
  EXPECT_EQ(C.dim[1], C_check.dim[1]);
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalLowrankNewDense) {
  //H LR New(D)
  if(beta != 0) GTEST_SKIP();
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const FRANK::LowRank B(BD, THRESHOLD);

  FRANK::Dense C_check(m, n);
  FRANK::Dense C = gemm(A, B, alpha, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, 0, transA, transB);

  // Check result
  EXPECT_EQ(C.dim[0], C_check.dim[0]);
  EXPECT_EQ(C.dim[1], C_check.dim[1]);
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalHierarchicalNewHierarchical) {
  //H H New(H)
  if(beta != 0) GTEST_SKIP();
  const FRANK::Dense AD(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const FRANK::Hierarchical A(FRANK::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);
  const FRANK::Dense BD(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const FRANK::Hierarchical B(FRANK::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, FRANK::AdmisType::PositionBased);

  FRANK::Dense C_check(m, n);
  FRANK::Hierarchical C = gemm(A, B, alpha, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, 0, transA, transB);

  // Check result
  EXPECT_EQ(FRANK::get_n_rows(C), FRANK::get_n_rows(C_check));
  EXPECT_EQ(FRANK::get_n_cols(C), FRANK::get_n_cols(C_check));
  const double error = FRANK::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

INSTANTIATE_TEST_SUITE_P(GEMM, GEMMTests,
                         testing::Combine(testing::Values(16, 32),
                                          testing::Values(16, 32),
                                          testing::Values(16, 32),
                                          testing::Values(-1.0, 0.0, 1.0),
                                          testing::Values(-1.0, 0.0, 1.0),
                                          testing::Values(true, false),
                                          testing::Values(true, false)
                                          ));
