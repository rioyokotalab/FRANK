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

  const int64_t N = argc > 1 ? atoi(argv[1]) : 128;
  const int64_t nleaf = argc > 2 ? atoi(argv[2]) : 32;
  const double eps = argc > 3 ? atof(argv[3]) : 1e-6;
  const int64_t admis = argc > 4 ? atoi(argv[4]) : 0;
  const int64_t nblocks = 2;
  const std::vector<std::vector<double>> randx{ get_sorted_random_vector(N) };

  print("Generate hicma laplace 1D.");
  timing::start("Hierarchical compression");
  const Hierarchical A(laplacend, randx, N, N, nleaf, eps, admis, nblocks, nblocks, AdmisType::PositionBased);
  timing::stopAndPrint("Hierarchical compression");
  write_JSON(A, std::string("laplace1d-") + std::to_string(N) + std::string("-") +
             std::to_string(nleaf) + std::string("-") +
             std::to_string(eps) + std::string("-") +
             std::to_string(admis) + std::string(".json"));
  timing::stopAndPrint("Overall");
  return 0;
}
