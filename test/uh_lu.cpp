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
  Dense D(laplace1d, randx, N, N);

  start("Verification");
  print("Compression Accuracy");
  print("H Rel. L2 Error", l2_error(D, H), false);
  print("UH Rel. L2 Error", l2_error(A, H), false);
  stop("Verification");
  return 0;
}
