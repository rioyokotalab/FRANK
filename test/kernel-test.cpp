#include "hicma/hicma.h"

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>
#include <iostream>
#include <fstream>

using namespace hicma;
using namespace std;

int main(int argc, char** argv) {
  timing::start("Overall");
  hicma::initialize();
  
  int64_t nleaf = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t N = atoi(argv[3]);
  int64_t admis = atoi(argv[4]);
  int64_t basis = 0;
  int64_t nblocks = N / nleaf;
  assert(basis == NORMAL_BASIS || basis == SHARED_BASIS);

  std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};

  print("Generate hicma laplace 1D.");
  timing::start("Hierarchical compression");
  Hierarchical A(laplacend, randx, N, N, rank, nleaf, admis,
               nblocks, nblocks, basis);
  timing::stop("Hierarchical compression");
  printXML(A, std::string("laplace1d-") + std::to_string(N) + std::string("-") + std::to_string(nleaf) + std::string("-") +
           std::to_string(rank) + std::string("-") + std::to_string(admis) + std::string(".xml"));

  print("Generate hicma Cauchy 2D.");
  std::vector<std::vector<double>> randx1{get_sorted_random_vector(N), get_sorted_random_vector(N)};
  Hierarchical A1(cauchy2d, randx1, N, N, rank, nleaf, admis,
               nblocks, nblocks, basis);  
  printXML(A1, std::string("cauchy2d-") + std::to_string(N) + std::string("-") + std::to_string(nleaf) + std::string("-") +
           std::to_string(rank) + std::string("-") + std::to_string(admis) + std::string(".xml"));

  print("Generate stars-h 3D exponential.");
  double beta = 0.1;
  double nu = 0.5;//in matern, nu=0.5 exp (half smooth), nu=inf sqexp (inifinetly smooth)
  //nu is only used in matern kernel
  //double noise = 1.e-2; // did not work for 10M in Lorapo
  double noise = 1.e-1;
  double sigma = 1.0;
  double wave_k = 1.0;
  starsh::exp_kernel_prepare(N, beta, nu, noise, sigma, 3);

  Hierarchical A2(starsh::exp_kernel_fill, randx, N, N, rank, nleaf, admis,
                 nblocks, nblocks, basis);
  printXML(A2, std::string("exp-3d-") + std::to_string(N) + std::string("-") + std::to_string(nleaf) + std::string("-") +
           std::to_string(rank) + std::string("-") + std::to_string(admis) + std::string(".xml"));
  starsh::exp_kernel_cleanup();


  starsh::exp_kernel_prepare(N, beta, nu, noise, sigma, 2);

  Hierarchical A3(starsh::exp_kernel_fill, randx, N, N, rank, nleaf, admis,
                  nblocks, nblocks, basis);
  printXML(A3, std::string("exp-2d-") + std::to_string(N) + std::string("-") + std::to_string(nleaf) + std::string("-") +
           std::to_string(rank) + std::string("-") + std::to_string(admis) + std::string(".xml"));
  starsh::exp_kernel_cleanup();
  
  return 0;
}
