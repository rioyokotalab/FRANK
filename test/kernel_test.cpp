#include "hicma/hicma.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>
#include <utility>
#include <cmath>


using namespace hicma;

TEST(MatrixKernelTest, ZeroKernel) {
  hicma::initialize();
  int64_t n = 3;

  Dense<double> A(zeros, n,n);
  for (int64_t i=0; i<n; ++i)
    for (int64_t j=0; j<n; ++j)
      EXPECT_EQ(A(i,j), 0);
}

TEST(MatrixKernelTest, ArangeKernel) {
  hicma::initialize();
  int64_t m = 5;
  int64_t n = 7;
  
  Dense<double> A(arange, m, n);
  for (int64_t i=0; i<m; ++i)
    for (int64_t j=0; j<n; ++j)
      EXPECT_EQ(A(i,j), i*n + j);

  /*Dense<float> Af (m,n);
  std::vector<Dense<float>> submatrices = Af.split(2,2);
  for (size_t i=0; i<submatrices.size(); ++i) {
    arange_kernel.apply(submatrices[i], i>1?3:0, i%2?4:0);
  }
  for (int64_t i=0; i<m; ++i)
    for (int64_t j=0; j<n; ++j)
      EXPECT_EQ(Af(i,j), i*n + j);

  std::vector<std::vector<double>> range = arange_kernel.get_coords_range(IndexRange(0, m));
  EXPECT_EQ(range.size(), 0);
  range = arange_kernel.get_coords_range(IndexRange(0, n));
  EXPECT_EQ(range.size(), 0);*/
}

TEST(MatrixKernelTest, IdentityKernel) {
  hicma::initialize();
  int64_t m = 8;
  int64_t n = 6;
  
  Dense<float> Af(identity, m, n);
  for (int64_t i=0; i<m; ++i)
    for (int64_t j=0; j<n; ++j)
      EXPECT_EQ(Af(i,j), i==j?1:0);

  /*
  Dense<double> A(m,n);
  std::vector<Dense<double>> submatrices = A.split(2,2);
  for (size_t i=0; i<submatrices.size(); ++i) {
    identity_kernel.apply(submatrices[i], i>1?4:0, i%2?3:0);
  }
  for (int64_t i=0; i<m; ++i)
    for (int64_t j=0; j<n; ++j)
      EXPECT_EQ(A(i,j), i==j?1:0);

  std::vector<std::vector<float>> range = identity_kernel.get_coords_range(IndexRange(0, m));
  EXPECT_EQ(range.size(), 0);
  range = identity_kernel.get_coords_range(IndexRange(0, n));
  EXPECT_EQ(range.size(), 0);*/
}

TEST(MatrixKernelTest, RandomNormalKernel) {
  hicma::initialize();
  int64_t n = 10;
  
  /* generate from double */

  Dense<double> A(random_normal, n, n);
  double sum = 0;
  for (int64_t i=0; i<n; ++i){
    for (int64_t j=0; j<n; ++j) {
      sum += A(i,j);
    }
  }
  sum /= (n*n);
  // TODO improve this check
  EXPECT_LT(sum, 0.25);
  EXPECT_GT(sum, -0.25);


  Dense<float> Af(random_normal, n ,n);
  for (int64_t i=0; i<n; ++i){
    for (int64_t j=0; j<n; ++j) {
      EXPECT_FLOAT_EQ(A(i,j), Af(i,j));
    }
  }

  // check different seed
  /*
  RandomNormalKernel<double> seeded_normal_kernel(5);
  Dense<double> B(n,n);
  seeded_normal_kernel.apply(B);
  Dense<float> Bf(n,n);
  seeded_normal_kernel.apply(Bf);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_FLOAT_EQ(B(i,j), Bf(i,j));
      EXPECT_NE(Af(i,j), Bf(i,j));
      EXPECT_NE(A(i,j), B(i,j));
    }
  }
  */

  /* generate from float */

  /*Dense<double> C(n,n);
  RandomNormalKernel<float> normal_kernel_f;
  normal_kernel_f.apply(C);
  sum = 0;
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      sum += C(i,j);
    }
  }
  sum /= (n*n);
  // TODO improve this check
  EXPECT_LT(sum, 0.25);
  EXPECT_GT(sum, -0.25);

  Dense<float> Cf(n,n);
  normal_kernel_f.apply(Cf);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_EQ(C(i,j), Cf(i,j));
      EXPECT_NE(C(i,j), A(i,j));
    }
  }

  // random seed
  Dense<double> D(n,n);
  normal_kernel_f.apply(D, false);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_NE(C(i,j), D(i,j));
    }
  }

  std::vector<std::vector<double>> range = normal_kernel.get_coords_range(IndexRange(0, n));
  EXPECT_EQ(range.size(), 0);
  std::vector<std::vector<float>> range_f = normal_kernel_f.get_coords_range(IndexRange(0, n));
  EXPECT_EQ(range_f.size(), 0);*/
}

TEST(MatrixKernelTest, RandomUniformKernel) {
  hicma::initialize();
  int64_t n = 10;
  
  /* generate from double */

  Dense<double> A(random_uniform, n ,n);
  for (int64_t i=0; i<n; ++i){
    for (int64_t j=0; j<n; ++j) {
      EXPECT_LE(A(i,j), 1);
      EXPECT_GE(A(i,j), 0);
    }
  }

  Dense<float> Af(random_uniform, n, n);
  for (int64_t i=0; i<n; ++i){
    for (int64_t j=0; j<n; ++j) {
      EXPECT_FLOAT_EQ(A(i,j), Af(i,j));
    }
  }

  /*
  // check different seed
  RandomUniformKernel<double> seeded_uniform_kernel(5);
  Dense<double> B(n,n);
  seeded_uniform_kernel.apply(B);
  Dense<float> Bf(n,n);
  seeded_uniform_kernel.apply(Bf);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_FLOAT_EQ(B(i,j), Bf(i,j));
      EXPECT_NE(Af(i,j), Bf(i,j));
      EXPECT_NE(A(i,j), B(i,j));
    }
  }*/

  /* generate from float */
  /*
  Dense<double> C(n,n);
  RandomUniformKernel<float> uniform_kernel_f;
  uniform_kernel_f.apply(C);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_LE(C(i,j), 1);
      EXPECT_GE(C(i,j), 0);
    }
  }

  Dense<float> Cf(n,n);
  uniform_kernel_f.apply(Cf);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_EQ(C(i,j), Cf(i,j));
      EXPECT_NE(C(i,j), A(i,j));
    }
  }

  // random seed
  Dense<double> D(n,n);
  uniform_kernel_f.apply(D, false);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_NE(C(i,j), D(i,j));
    }
  }

  std::vector<std::vector<double>> range = uniform_kernel.get_coords_range(IndexRange(0, n));
  EXPECT_EQ(range.size(), 0);
  std::vector<std::vector<float>> range_f = uniform_kernel_f.get_coords_range(IndexRange(0, n));
  EXPECT_EQ(range_f.size(), 0);*/
}

TEST(MatrixKernelTest, Cauchy2dKernel) {
  hicma::initialize();
  int64_t n = 25;
  double error = 1e-2;

  /* double parameters */
  std::vector<std::vector<double>> params;
  params.push_back(get_sorted_random_vector<double>(n));
  params.push_back(get_sorted_random_vector<double>(n));

  Dense<double> A(cauchy2d, params, n, n);
  Dense<float> Af(cauchy2d, params, n, n);
  for (int64_t i=0; i<n; ++i){
    for (int64_t j=0; j<n; ++j){
      double d = params[0][i] - params[1][j] + error;
      d = 1/d;
      EXPECT_DOUBLE_EQ(A(i,j), d);
      EXPECT_FLOAT_EQ(Af(i,j), d);
    }
  }
  
  /* float parameters */
  /*
  std::vector<std::vector<float>> params_f;
  params_f.push_back(get_sorted_random_vector<float>(n));
  params_f.push_back(get_sorted_random_vector<float>(n));
  std::vector<std::vector<float>> init_params (params_f);
  Cauchy2dKernel<float> cauchy_kernel_f(std::move(init_params));
  EXPECT_EQ(init_params.size(), 0);

  Dense<double> B(n,n);
  Dense<float> Bf(n,n);
  cauchy_kernel_f.apply(B);
  cauchy_kernel_f.apply(Bf);
  for (int64_t i=0; i<n; ++i){
    for (int64_t j=0; j<n; ++j){
      float d = params_f[0][i] - params_f[1][j] + error;
      d = 1/d;
      EXPECT_DOUBLE_EQ(B(i,j), d);
      EXPECT_FLOAT_EQ(Bf(i,j), d);
    }
  }

  Dense<double> C(n,n);
  std::vector<Dense<double>> submatrices = C.split(2,2);
  for (size_t i=0; i<submatrices.size(); ++i) {
    cauchy_kernel.apply(submatrices[i], i>1?13:0, i%2?13:0);
  }
  for (int64_t i=0; i<n; ++i)
    for (int64_t j=0; j<n; ++j)
      EXPECT_EQ(A(i,j), C(i,j));*/
}

TEST(MatrixKernelTest, LaplacendKernel) {
  hicma::initialize();
  int64_t n = 50;
  double error = 1e-3;

  /* double parameters */
  std::vector<std::vector<double>> params;
  params.push_back(get_sorted_random_vector<double>(n));

  Dense<double> A(laplacend, params, n, n);
  Dense<float> Af(laplacend, params,n ,n);
  for (int64_t i=0; i<n; ++i){
    for (int64_t j=0; j<n; ++j){
      double d = (params[0][i] - params[0][j]) * (params[0][i] - params[0][j]);
      d = 1/(std::sqrt(d) + error);
      EXPECT_DOUBLE_EQ(A(i,j), d);
      EXPECT_FLOAT_EQ(Af(i,j), d);
    }
  }

  /* float parameters */
  /*
  std::vector<std::vector<float>> params_f;
  params_f.push_back(get_sorted_random_vector<float>(n));
  std::vector<std::vector<float>> init_params (params_f);
  LaplacendKernel<float> laplace_kernel_f(std::move(init_params));
  EXPECT_EQ(init_params.size(), 0);

  Dense<double> B(n,n);
  Dense<float> Bf(n,n);
  laplace_kernel_f.apply(B);
  laplace_kernel_f.apply(Bf);
  for (int64_t i=0; i<n; ++i){
    for (int64_t j=0; j<n; ++j){
      float d = (params_f[0][i] - params_f[0][j]) * (params_f[0][i] - params_f[0][j]);
      double div = 1/(std::sqrt(d) + error);
      d = 1/(std::sqrt(d) + error);
      EXPECT_DOUBLE_EQ(B(i,j), div);
      EXPECT_FLOAT_EQ(Bf(i,j), d);
    }
  }

  Dense<double> C(n,n);
  std::vector<Dense<double>> submatrices = C.split(2,2);
  for (size_t i=0; i<submatrices.size(); ++i) {
    laplace_kernel.apply(submatrices[i], i>1?25:0, i%2?25:0);
  }
  for (int64_t i=0; i<n; ++i)
    for (int64_t j=0; j<n; ++j)
      EXPECT_EQ(A(i,j), C(i,j));*/
}

TEST(MatrixKernelTest, HelmholtzndKernel) {
  hicma::initialize();
  int64_t n = 10;
  double error = 1e-3;

  /* double parameters */
  std::vector<std::vector<double>> params;
  params.push_back(get_sorted_random_vector<double>(n));

  Dense<double> A(helmholtznd, params, n, n);
  Dense<float> Af(helmholtznd, params, n, n);
  for (int64_t i=0; i<n; ++i){
    for (int64_t j=0; j<n; ++j){
      double d = (params[0][i] - params[0][j]) * (params[0][i] - params[0][j]);
      d = std::exp(-1.0*d)/(std::sqrt(d) + error);
      EXPECT_DOUBLE_EQ(A(i,j), d);
      EXPECT_FLOAT_EQ(Af(i,j), d);
    }
  }

  /* float parameters */
  /*
  std::vector<std::vector<float>> params_f;
  params_f.push_back(get_sorted_random_vector<float>(n));
  std::vector<std::vector<float>> init_params (params_f);
  HelmholtzndKernel<float> helmholtz_kernel_f(std::move(init_params));
  EXPECT_EQ(init_params.size(), 0);

  Dense<double> B(n,n);
  Dense<float> Bf(n,n);
  helmholtz_kernel_f.apply(B);
  helmholtz_kernel_f.apply(Bf);
  for (int64_t i=0; i<n; ++i){
    for (int64_t j=0; j<n; ++j){
      float d = (params_f[0][i] - params_f[0][j]) * (params_f[0][i] - params_f[0][j]);
      double div = std::exp(-1.0*d)/(std::sqrt(d) + error);
      d = std::exp(-1.0*d)/(std::sqrt(d) + error);
      EXPECT_DOUBLE_EQ(B(i,j), div);
      EXPECT_FLOAT_EQ(Bf(i,j), d);
    }
  }

  Dense<double> C(n,n);
  std::vector<Dense<double>> submatrices = C.split(2,2);
  for (size_t i=0; i<submatrices.size(); ++i) {
    helmholtz_kernel.apply(submatrices[i], i>1?5:0, i%2?5:0);
  }
  for (int64_t i=0; i<n; ++i)
    for (int64_t j=0; j<n; ++j)
      EXPECT_EQ(A(i,j), C(i,j));*/
}
