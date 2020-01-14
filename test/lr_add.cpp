#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main(int argc, char** argv) {
  yorel::multi_methods::initialize();
  int N = 2048;
  int rank = 128;
  std::vector<double> randx(2*N);
  for (int i=0; i<2*N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
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
