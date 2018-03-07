#include "mpi_utils.h"
#include "functions.h"
#include "hblas.h"
#include "print.h"
#include "timer.h"

using namespace hicma;

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
