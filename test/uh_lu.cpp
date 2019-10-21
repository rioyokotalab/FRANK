#include "hicma/uniform_hierarchical.h"
#include "hicma/functions.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main(int argc, char const *argv[])
{
  yorel::multi_methods::initialize();
  int N = 128;
  int nleaf = 32;
  int rank = 8;
  int admis = 0;
  int ni_level = 2, nj_level = 2;
  std::vector<double> randx(N);
  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  UniformHierarchical H(
    laplace1d, randx, N, N, rank, nleaf, admis, ni_level, nj_level);
  return 0;
}
