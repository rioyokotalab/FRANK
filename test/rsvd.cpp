#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/functions.h"
#include "hicma/operations/norm.h"
#include "hicma/util/l2_error.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main(int argc, char** argv) {
  yorel::multi_methods::initialize();
  int N = 32;
  int rank = 16;
  std::vector<double> randx(2*N);
  for (int i=0; i<2*N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  print("Time");
  start("Init matrix");
  Dense D(laplace1d, randx, N, N-2, 0, N);
  stop("Init matrix");
  start("Randomized SVD");
  LowRank LR(D, rank);
  stop("Randomized SVD");
  print("Accuracy");
  print("Rel. L2 Error", l2_error(D, LR), false);
  return 0;
}
