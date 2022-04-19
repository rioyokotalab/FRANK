#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include <limits>

#include "hicma/hicma.h"
#include "gtest/gtest.h"

class GEMMTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t, double, double, bool, bool>> {
 protected:
  void SetUp() override {
    hicma::initialize();
    hicma::setGlobalValue("HICMA_LRA", "rounded_addition");
    std::tie(m, k, n, alpha, beta, transA, transB) = GetParam();
    randx_A = { hicma::get_sorted_random_vector(4 * std::max(m, k)) };
    randx_B = { hicma::get_sorted_random_vector(4 * std::max(k, n)) };
    randx_C = { hicma::get_sorted_random_vector(4 * std::max(m, n)) };
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

static void naive_gemm(const hicma::Dense &A, const hicma::Dense &B, hicma::Dense &C,
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
  const hicma::Dense A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Dense B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  hicma::Dense C(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(C);
  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  
  naive_gemm(A, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, EPSILON);
}

TEST_P(GEMMTests, DenseDenseLowrank) {
  //D D LR
  const hicma::Dense A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Dense B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  hicma::Dense CD(hicma::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  hicma::Dense C_check(CD);
  hicma::LowRank C(CD, THRESHOLD);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseDenseHierarchical) {
  //D D H
  const hicma::Dense A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Dense B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  hicma::Dense CD(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(CD);
  hicma::Hierarchical C(hicma::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseLowrankDense) {
  //D LR D
  const hicma::Dense A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const hicma::LowRank B(BD, THRESHOLD);
  
  hicma::Dense C(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(C);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseLowrankLowrank) {
  //D LR LR
  const hicma::Dense A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const hicma::LowRank B(BD, THRESHOLD);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  hicma::Dense C_check(CD);
  hicma::LowRank C(CD, THRESHOLD);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseLowrankHierarchical) {
  //D LR LR
  const hicma::Dense A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const hicma::LowRank B(BD, THRESHOLD);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(CD);
  hicma::Hierarchical C(hicma::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseHierarchicalDense) {
  //D H D
  const hicma::Dense A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const hicma::Hierarchical B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::Dense C(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(C);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseHierarchicalLowrank) {
  //D H LR
  const hicma::Dense A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const hicma::Hierarchical B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  hicma::Dense C_check(CD);
  hicma::LowRank C(CD, THRESHOLD);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, DenseHierarchicalHierarchical) {
  //D H H
  const hicma::Dense A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const hicma::Hierarchical B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(CD);
  hicma::Hierarchical C(hicma::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(A, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankDenseDense) {
  //LR D D
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const hicma::LowRank A(AD, THRESHOLD);
  const hicma::Dense B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);

  hicma::Dense C(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(C);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankDenseLowrank) {
  //LR D LR
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const hicma::LowRank A(AD, THRESHOLD);
  const hicma::Dense B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  hicma::Dense C_check(CD);
  hicma::LowRank C(CD, THRESHOLD);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankDenseHierarchical) {
  //LR D H
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const hicma::LowRank A(AD, THRESHOLD);
  const hicma::Dense B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(CD);
  hicma::Hierarchical C(hicma::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankLowrankDense) {
  //LR LR D
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const hicma::LowRank A(AD, THRESHOLD);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const hicma::LowRank B(BD, THRESHOLD);

  hicma::Dense C(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(C);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankLowrankLowrank) {
  //LR LR LR
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const hicma::LowRank A(AD, THRESHOLD);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const hicma::LowRank B(BD, THRESHOLD);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  hicma::Dense C_check(CD);
  hicma::LowRank C(CD, THRESHOLD);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankLowrankHierarchical) {
  //LR LR H
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const hicma::LowRank A(AD, THRESHOLD);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const hicma::LowRank B(BD, THRESHOLD);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(CD);
  hicma::Hierarchical C(hicma::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankHierarchicalDense) {
  //LR H D
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const hicma::LowRank A(AD, THRESHOLD);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const hicma::Hierarchical B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::Dense C(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(C);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankHierarchicalLowrank) {
  //LR H LR
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const hicma::LowRank A(AD, THRESHOLD);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const hicma::Hierarchical B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  hicma::Dense C_check(CD);
  hicma::LowRank C(CD, THRESHOLD);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, LowrankHierarchicalHierarchical) {
  //LR H H
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k, 0, 2 * std::max(m, k));
  const hicma::LowRank A(AD, THRESHOLD);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const hicma::Hierarchical B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(CD);
  hicma::Hierarchical C(hicma::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalDenseDense) {
  //H D D
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Hierarchical A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);
  const hicma::Dense B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);

  hicma::Dense C(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(C);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalDenseLowrank) {
  //H D LR
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Hierarchical A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);
  const hicma::Dense B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  hicma::Dense C_check(CD);
  hicma::LowRank C(CD, THRESHOLD);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalDenseHierarchical) {
  //H D H
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Hierarchical A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);
  const hicma::Dense B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(CD);
  hicma::Hierarchical C(hicma::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, B, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalLowrankDense) {
  //H LR D
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Hierarchical A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const hicma::LowRank B(BD, THRESHOLD);

  hicma::Dense C(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(C);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalLowrankLowrank) {
  //H LR LR
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Hierarchical A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const hicma::LowRank B(BD, THRESHOLD);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  hicma::Dense C_check(CD);
  hicma::LowRank C(CD, THRESHOLD);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalLowrankHierarchical) {
  //H LR H
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Hierarchical A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n, 0, 2 * std::max(n, k));
  const hicma::LowRank B(BD, THRESHOLD);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(CD);
  hicma::Hierarchical C(hicma::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalHierarchicalDense) {
  //H H D
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Hierarchical A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const hicma::Hierarchical B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::Dense C(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(C);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalHierarchicalLowrank) {
  //H H LR
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Hierarchical A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const hicma::Hierarchical B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n, 0, 2 * std::max(m, n));
  hicma::Dense C_check(CD);
  hicma::LowRank C(CD, THRESHOLD);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

TEST_P(GEMMTests, HierarchicalHierarchicalHierarchical) {
  //H H H
  const hicma::Dense AD(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k);
  const hicma::Hierarchical A(hicma::laplacend, randx_A, transA ? k : m, transA ? m : k,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);
  const hicma::Dense BD(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n);
  const hicma::Hierarchical B(hicma::laplacend, randx_B, transB ? n : k, transB ? k : n,
                              nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::Dense CD(hicma::laplacend, randx_C, m, n);
  hicma::Dense C_check(CD);
  hicma::Hierarchical C(hicma::laplacend, randx_C, m, n,
                        nleaf, THRESHOLD, admis, nblocks, nblocks, hicma::AdmisType::PositionBased);

  hicma::gemm(A, B, C, alpha, beta, transA, transB);
  naive_gemm(AD, BD, C_check, alpha, beta, transA, transB);

  // Check result
  const double error = hicma::l2_error(C_check, C);
  EXPECT_LE(error, 10 * THRESHOLD);
}

INSTANTIATE_TEST_SUITE_P(InPlaceGEMM, GEMMTests,
                         testing::Combine(testing::Values(16, 32),
                                          testing::Values(16, 32),
                                          testing::Values(16, 32),
                                          testing::Values(-1.0, 0.0, 1.0),
                                          testing::Values(-1.0, 0.0, 1.0),
                                          testing::Values(true, false),
                                          testing::Values(true, false)
                                          ));
