#include "hicma/hicma.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>


using namespace hicma;

TEST(LowRankTest, Constructors) {
  hicma::initialize();
  int64_t n = 15;
  int64_t rank = 4;

  LowRank<float> Af;
  EXPECT_EQ(Af.dim[0], 0);
  EXPECT_EQ(Af.dim[1], 0);
  EXPECT_EQ(Af.rank, 0);
  
  // 3 matrices
  Dense<double> U (n, rank);
  Dense<double> S (rank, rank);
  Dense<double> V (rank, n);
  LowRank<double> A(U, S, V);
  EXPECT_EQ(A.dim[0], n);
  EXPECT_EQ(A.dim[1], n);
  EXPECT_EQ(A.rank, rank);
  // check for copy
  S(0, 0) = 1;
  EXPECT_EQ(A.S(0,0), 0);

  LowRank<double> B(U, S, V, false);
  // check for copy
  S(0, 0) = 1;
  EXPECT_EQ(B.S(0,0), 1);

  LowRank<float> Bf(Dense<float>(n, rank), Dense<float>(rank, rank), Dense<float>(rank, n+3));
  EXPECT_EQ(Bf.dim[0], n);
  EXPECT_EQ(Bf.dim[1], n+3);
  EXPECT_EQ(Bf.rank, rank);

  // move costructor from MatrixProxy
  LowRank<double> C(std::move(MatrixProxy(B)));
  EXPECT_EQ(C.dim[0], n);
  EXPECT_EQ(C.dim[1], n);
  EXPECT_EQ(C.rank, rank);
  EXPECT_EQ(C.S(0,0), 1);

}

// Check whether the LowRank(Dense) constructor works correctly.
TEST(LowRankTest, ContructorDense) {
  hicma::initialize();
  int64_t n = 1024;
  int64_t rank = 15;

  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector<double>(2*n)};
  Dense<double> A(laplacend, randx_A, n, n, 0, n);

  // double precision
  LowRank<double> LR(A, rank);
  double error = l2_error(A, LR);
  EXPECT_LT(error, 1e-8);

  // single precision
  Dense<float> Af(laplacend, randx_A, n, n, 0, n);
  LowRank<float> LRf(Af, rank);
  error = l2_error(Af, LRf);
  EXPECT_LT(error, 1e-5);
}
