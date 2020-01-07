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
  int ni_level = atoi(argv[1]), nj_level = atoi(argv[1]);
  int N = atoi(argv[2]);
  int nleaf = atoi(argv[3]);
  int rank = atoi(argv[4]);
  int admis = 0;
  std::vector<double> randx(N);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  UniformHierarchical A(
    laplace1d, randx, N, N, rank, nleaf, admis, ni_level, nj_level);
  Hierarchical H(
    laplace1d, randx, N, N, rank, nleaf, admis, ni_level, nj_level);
  Hierarchical D(laplace1d, randx, N, N, rank, nleaf, N/ni_level, ni_level, nj_level);
  Dense rand(random_normal, randx, N, N);

  start("Verification");
  print("Compression Accuracy");
  print("H Rel. L2 Error", l2_error(D, H), false);
  print("UH Rel. L2 Error", l2_error(A, H), false);
  stop("Verification");

  Dense test1(N, N);
  gemm(A, rand, test1, 1, 0);
  Dense test2(N, N);
  gemm(H, rand, test2, 1, 0);
  Dense test3(N, N);
  gemm(D, rand, test3, 1, 0);

  print("UH-H diff", l2_error(test1, test2), false);
  print("UH-D diff", l2_error(test1, test3), false);
  print("H-D diff", l2_error(test2, test3), false);

  UniformHierarchical L, U;
  std::tie(L, U) = getrf(A);

  Hierarchical LH, UH;
  std::tie(LH, UH) = getrf(H);

  print("L UH-H diff", l2_error(L(1, 0), LH(1, 0)), false);
  print("U UH-H diff", l2_error(U(0, 1), UH(0, 1)), false);


  return 0;
}
