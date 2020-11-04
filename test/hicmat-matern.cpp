#include "hicma/hicma.h"

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>
#include <iostream>

#include <starsh.h>

using namespace hicma;
using namespace std;

int main(int argc, char** argv) {
  timing::start("Overall");
  hicma::initialize();
  
  // int64_t nleaf = atoi(argv[1]);
  // int64_t rank = atoi(argv[2]);
  // int64_t admis = atoi(argv[3]);
  // int64_t N = atoi(argv[4]);


  int64_t nleaf = 1024;
  int64_t rank = 16;
  int64_t admis = 0;
  int64_t N = 8192;

  int64_t basis = 0;
  int64_t nblocks = N / nleaf;
  assert(basis == NORMAL_BASIS || basis == SHARED_BASIS);

  /* Default parameters for statistics */
  double beta = 0.1;
  double nu = 0.5;//in matern, nu=0.5 exp (half smooth), nu=inf sqexp (inifinetly smooth)
  //nu is only used in matern kernel
  //double noise = 1.e-2; // did not work for 10M in Lorapo
  double noise = 1.e-1;
  double sigma = 1.0;
  double wave_k = 1.0;
  int add_diag = 27000;

  std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};

  starsh::matern_kernel_prepare(N, beta, nu, noise, sigma, wave_k, add_diag);

  Hierarchical(starsh::matern_kernel_fill, randx, N, N, rank, nleaf, admis,
               nblocks, nblocks, basis);
  
  return 0;
}
