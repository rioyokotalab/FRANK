#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/low_rank_shared.h"
#include "hicma/hierarchical.h"
#include "hicma/uniform_hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations/norm.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

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
  Hierarchical DH(D, ni_level, nj_level);
  LowRankShared& LRS = static_cast<LowRankShared&>(*A(0, 1));
  LowRank& LR = static_cast<LowRank&>(*H(0, 1));

  start("Verification");
  double l2 = norm(D);
  double diff1 = norm(D - Dense(H));
  double diff2 = norm(Dense(A) - Dense(H));
  stop("Verification");
  print("Compression Accuracy");
  print("H Rel. L2 Error", std::sqrt(diff1/l2), false);
  print("UH Rel. L2 Error", std::sqrt(diff2/l2), false);
  return 0;
}
