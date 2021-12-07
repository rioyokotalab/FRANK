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

  int64_t nleaf = argc > 1 ? atoi(argv[1]) : 32;
  int64_t rank = argc > 2 ? atoi(argv[2]) : 16;
  int64_t N = argc > 3 ? atoi(argv[3]) : 256;
  int64_t admis = argc > 4 ? atoi(argv[4]) : 0;
  int64_t nblocks = N / nleaf;

  std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};

  print("Generate hicma laplace 1D.");
  timing::start("Hierarchical compression");
  Hierarchical A(laplacend, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  timing::stop("Hierarchical compression");
  write_JSON(A, std::string("laplace1d-") + std::to_string(N) + std::string("-") + std::to_string(nleaf) + std::string("-") +
           std::to_string(rank) + std::string("-") + std::to_string(admis) + std::string(".json"));

  print("Generate hicma Cauchy 2D.");
  std::vector<std::vector<double>> randx1{get_sorted_random_vector(N), get_sorted_random_vector(N)};
  Hierarchical A1(cauchy2d, randx1, N, N, rank, nleaf, admis, nblocks, nblocks);
  write_JSON(A1, std::string("cauchy2d-") + std::to_string(N) + std::string("-") + std::to_string(nleaf) + std::string("-") +
           std::to_string(rank) + std::string("-") + std::to_string(admis) + std::string(".json"));

  return 0;
}
