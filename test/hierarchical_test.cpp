#include "hicma/hicma.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <utility>


using namespace hicma;

TEST(HierarchicalTest, StandardConstructors) {
  hicma::initialize();
  int64_t n=4;

  Hierarchical<double> A;
  EXPECT_EQ(A.dim[0], 0);
  EXPECT_EQ(A.dim[1], 0);

  Hierarchical<float> Af(n);
  EXPECT_EQ(Af.dim[0], 4);
  EXPECT_EQ(Af.dim[1], 1);

  Hierarchical<double> B(n, n);
  EXPECT_EQ(B.dim[0], 4);
  EXPECT_EQ(B.dim[1], 4);
}


TEST(HierarchicalTest, ConstructorKernel) {
  int64_t n = 512;
  int64_t nleaf = 32;
  int64_t rank = 16;
  int64_t nblocks = 2;
  double admis = 0;
  vec2d<double> randx{get_sorted_random_vector<double>(n)};

  // TODO add construction from function

  /* DOUBLE PRECISION */

  // construct from kernel
  Hierarchical<double> A(laplacend, randx, n, n, rank, nleaf, admis, nblocks, nblocks);
  EXPECT_EQ(A.dim[0], 2);
  EXPECT_EQ(A.dim[1], 2);
  Dense<double> check(laplacend, randx, n, n);
  double error_kernel = l2_error(A, check);
  EXPECT_LT(error_kernel, 1e-12);

  // construct from function
  //Hierarchical<double> B(laplacend, n, n, rank, nleaf, admis, nblocks, nblocks, randx);
  //double error_func = l2_error(B, check);
  //EXPECT_EQ(error_kernel, error_func);

  /* SINGLE PRECISION */

  // construct from kernel
  /*LaplacendKernel<double> kernel(randx);
  Hierarchical<float> Af(std::move(kernel), n, n, rank, nleaf);
  EXPECT_EQ(Af.dim[0], 2);
  EXPECT_EQ(Af.dim[1], 2);
  Dense<float> check_f(LaplacendKernel<double>(randx), n, n);
  Dense<float> check_move_f(kernel, n, n);
  error_kernel = l2_error(Af, check_f);
  EXPECT_LT(error_kernel, 1e-6);
  // check that the kernel has been moved
  double error_move = l2_error(Af, check_move_f);
  EXPECT_GT(error_move, 1.0);*/

  // construct from function
  Dense<float> check_f(laplacend, randx, n, n);
  Hierarchical<float> Bf(laplacend, randx, n, n, rank, nleaf, admis, nblocks, nblocks);
  double error_func = l2_error(Bf, check_f);
  EXPECT_LT(error_func, 1e-6);
}


TEST(HierarchicalTest, ConstructorMatrix) {
  int64_t n = 256;
  int64_t nleaf = 32;
  int64_t rank = 16;
  double admis = 1;
  vec2d<double> randx = {get_sorted_random_vector<double>(n)};

  /* DOUBLE PRECISION */

  // construct from Dense
  Dense<double> check(laplacend, randx, n, n);
  Dense<double> test(check);
  Hierarchical<double> A(std::move(test), rank, nleaf, admis);
  EXPECT_EQ(A.dim[0], 2);
  EXPECT_EQ(A.dim[1], 2);
  double error = l2_error(A, check);
  EXPECT_LT(error, 1e-15);

  // move costructor from MatrixProxy
  Hierarchical<double> B(std::move(MatrixProxy(A)));
  EXPECT_EQ(B.dim[0], 2);
  EXPECT_EQ(B.dim[1], 2);
  double error_move = l2_error(B, check);
  EXPECT_EQ(error, error_move);

  /* SINGLE PRECISION */

  // construct from Dense
  int64_t nblocks = 2;
  Hierarchical<float> Af(Dense<float> (laplacend, randx, n, n), rank , nleaf, admis, nblocks, nblocks);
  Dense<float> check_f(laplacend, randx, n, n);
  EXPECT_EQ(Af.dim[0], 2);
  EXPECT_EQ(Af.dim[1], 2);
  error = l2_error(Af, check_f);
  EXPECT_LT(error, 1e-7);
 
  // move costructor from MatrixProxy
  Hierarchical<float> Bf(std::move(MatrixProxy(Af)));
  EXPECT_EQ(Bf.dim[0], 2);
  EXPECT_EQ(Bf.dim[1], 2);
  error_move = l2_error(Bf, check_f);
  EXPECT_EQ(error, error_move);
}

TEST(HierarchicalTest, ConstructorFile) {
  // TODO construct from file?
}

// TODO test access operators