#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main(int argc, char const *argv[])
{
  yorel::multi_methods::initialize();
  int N = argc > 1 ? atoi(argv[1]) : 256;
  int nblocks = argc > 2 ? atoi(argv[2]) : 2;
  int rank = argc > 3 ? atoi(argv[3]) : 8;
  int admis = argc > 4 ? atoi(argv[4]) : 0;
  int nleaf = N/nblocks;
  std::vector<double> randx(N);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  start("Init matrix");
  UniformHierarchical A(
    laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  stop("Init matrix");
  start("Init matrix SVD");
  UniformHierarchical A_svd(
    laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks, true);
  stop("Init matrix SVD");
  Hierarchical H(
    laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical D(laplace1d, randx, N, N, rank, nleaf, N/nblocks, nblocks, nblocks);
  Dense rand(random_normal, randx, N, N);
  Dense x(random_uniform, randx, N, 1);
  Dense b(N, 1);
  gemm(A, x, b, 1, 1);

  start("Verification");
  print("Compression Accuracy");
  print("H Rel. L2 Error", l2_error(D, H), false);
  print("UH Rel. L2 Error", l2_error(D, A), false);
  stop("Verification");

  print("GEMM");
  start("GEMM");
  Dense test1(N, N);
  gemm(A, rand, test1, 1, 0);
  stop("GEMM");

  print("LU");
  start("UBLR LU");
  UniformHierarchical L, U;
  std::tie(L, U) = getrf(A);
  stop("UBLR LU");

  start("Verification");
  trsm(L, b,'l');
  trsm(U, b,'u');
  print("UH Rel. L2 Error", l2_error(x, b), false);
  stop("Verification");


  return 0;
}
