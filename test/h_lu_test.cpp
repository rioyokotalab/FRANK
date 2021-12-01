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
  Dense x(random_uniform, std::vector<std::vector<double>>(), n);
  Dense b(n);
  Dense D(laplacend, randx, n, n);
  Hierarchical A(laplacend, randx, n, n, rank, nleaf, admis, nblocks, nblocks);
  gemm(A, x, b, 1, 1);
  double compress_error = l2_error(A, D);

  Hierarchical L, U;
  std::tie(L, U) = getrf(A);
  trsm(L, b, TRSM_LOWER);
  trsm(U, b, TRSM_UPPER);
  double solve_error = l2_error(x, b);

  // Check result
  EXPECT_NEAR(compress_error, solve_error, compress_error);

}
