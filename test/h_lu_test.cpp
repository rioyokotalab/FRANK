#include "hicma/hicma.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>
#include <iostream>

using namespace hicma;

TEST(HierarchicalTest, h_lu) {
  hicma::initialize();
  int64_t n = 256;
  int64_t nleaf = 16;
  int64_t rank = 8;
  int64_t nblocks = 2;
  double admis = 0;
  std::vector<std::vector<double>> randx{get_sorted_random_vector(n)};
  timing::start("Total");
  Dense x(random_uniform, n);
  Dense b(n);
  Dense D(laplacend, randx, n, n);
  timing::start("Hierarchical compression");
  Hierarchical<double> A(laplacend, randx, n, n, rank, nleaf, admis, nblocks, nblocks);
  timing::stopAndPrint("Hierarchical compression");
  gemm(A, x, b, 1, 1);
  double compress_error = l2_error(A, D);

  Hierarchical L, U;
  timing::start("Factorization");
  std::tie(L, U) = getrf(A);
  timing::stopAndPrint("Factorization");
  timing::start("Solve");
  trsm(L, b, TRSM_LOWER);
  trsm(U, b, TRSM_UPPER);
  timing::stopAndPrint("Solve");
  double solve_error = l2_error(x, b);
  timing::stopAndPrint("Total");
  // Check result
  EXPECT_NEAR(compress_error, solve_error, compress_error);

  // single precision
  timing::start("F_Total");
  Dense<float> x_f(random_uniform<float>, n);
  Dense<float> b_f(n);
  Dense<float> D_f(laplacend<float>, randx, n, n);
  timing::start("F_Hierarchical compression");
  Hierarchical<float> A_f(laplacend<float>, randx, n, n, rank, nleaf, admis, nblocks, nblocks);
  timing::stopAndPrint("F_Hierarchical compression");
  gemm(A_f, x_f, b_f, 1, 1);
  compress_error = l2_error(A_f, D_f);

  Hierarchical<float> L_f, U_f;
  timing::start("F_Factorization");
  std::tie(L_f, U_f) = getrf(A_f);
  timing::stopAndPrint("F_Factorization");
  timing::start("F_Solve");
  trsm(L_f, b_f, TRSM_LOWER);
  trsm(U_f, b_f, TRSM_UPPER);
  timing::stopAndPrint("F_Solve");
  solve_error = l2_error(x_f, b_f);
  timing::stopAndPrint("F_Total");

  // Check result
  EXPECT_NEAR(compress_error, solve_error, 1e-5);

}
