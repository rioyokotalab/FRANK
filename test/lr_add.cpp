#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main() {
  yorel::multi_methods::initialize();
  int N = 2048;
  int rank = 128;
  std::vector<double> randx = get_sorted_random_vector(2*N);
  print("Time");
  timing::start("Init matrix");
  Dense D(laplace1d, randx, N, N, 0, N);
  LowRank A(D, rank);
  LowRank B(D, rank);
  timing::stopAndPrint("Init matrix");
  timing::start("LR Add");
  A += B;
  timing::stopAndPrint("LR Add", 2);
  print("Accuracy");
  print("Rel. L2 Error", l2_error(D+D, A), false);
  return 0;
}
