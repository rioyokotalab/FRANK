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
  print("Time");
  start("Init matrix");
  Hierarchical H(x,N,N,rank,nleaf,0,0,0,0,0);
  stop("Init matrix");
  start("LU decomposition");

  stop("LU decomposition");
  print2("-DGETRF");
  print2("-DTRSM");
  print2("-DGEMM");
  start("Forward substitution");

  stop("Forward substitution");
  start("Backward substitution");

  stop("Backward substitution");
  return 0;
}
