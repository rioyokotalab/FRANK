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
  timing::start("Total");
  std::vector<std::vector<double>> randx{get_sorted_random_vector(n)};
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

}
