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
  int nleaf = argc > 2 ? atoi(argv[2]) : 16;
  int rank = argc > 3 ? atoi(argv[3]) : 8;
  int nblocks = argc > 4 ? atoi(argv[4]) : 2;
  int admis = argc > 5 ? atoi(argv[5]) : 0;
  std::vector<double> randx(N);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  UniformHierarchical A(
    laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical H(
    laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical D(laplace1d, randx, N, N, rank, nleaf, N/nblocks, nblocks, nblocks);
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
