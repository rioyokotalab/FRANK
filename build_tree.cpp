#include "mpi_utils.h"
#include <algorithm>
#include "boost_any_wrapper.h"
#include <cmath>
#include <cstdlib>
#include "print.h"
#include "timer.h"
#include <vector>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 64;
  int rank = 8;
  int nleaf = 16;
  std::vector<double> x(N);
  for (int i=0; i<N; i++) {
    x[i] = drand48();
  }
  Hierarchical H(x,N,N,rank,nleaf);
  return 0;
}
