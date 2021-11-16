#include "hicma/hicma.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>


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

  Hierarchical L, U;
  std::tie(L, U) = getrf(A);
  trsm(L, b, TRSM_LOWER);
  trsm(U, b, TRSM_UPPER);

  // Check result
  for (int64_t i = 0; i < n; ++i) {
    EXPECT_NEAR(x(i, 0), b(i, 0), 1e-12);
  }

}
