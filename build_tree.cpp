#include "mpi_utils.h"
#include <algorithm>
#include "boost_any_wrapper.h"
#include <cmath>
#include <cstdlib>
#include "print.h"
#include "timer.h"
#include <vector>

using namespace hicma;

void laplace1d (
                std::vector<double>& data,
                std::vector<double>& x,
                const int& ni,
                const int& nj,
                const int& i_begin,
                const int& j_begin) {
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      data[i*nj+j] = 1 / (std::abs(x[i+i_begin] - x[j+j_begin]) + 1e-3);
    }
  }
}

int main(int argc, char** argv) {
  int N = 64;
  int rank = 8;
  int nleaf = 16;
  std::vector<double> x(N);
  for (int i=0; i<N; i++) {
    x[i] = drand48();
  }
  Hierarchical H(laplace1d, x, N, N, rank, nleaf, 0, 0, 0, 0, 0);
  return 0;
}
