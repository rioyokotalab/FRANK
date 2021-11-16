#include "hicma/hicma.h"

#include <cstdint>
#include <vector>


using namespace hicma;

int main() {
  hicma::initialize();
  int64_t N = 128;
  int64_t rank = 16;
  std::vector<std::vector<double>> randx{get_sorted_random_vector(2*N)};
  timing::start("Init matrix");
  Dense D(laplacend, randx, N, N, 0, N);
  LowRank A(D, rank);
  LowRank B(D, rank);
  LowRank AWork(A);
  timing::stop("Init matrix");

  print("LR Add Default");
  timing::start("LR Add Default");
  AWork += B;
  timing::stopAndPrint("LR Add Default", 2);
  print("Rel. L2 Error", l2_error(D+D, AWork), false);

  timing::start("Init matrix");
  AWork = A;
  timing::stop("Init matrix");
  print("LR Add Naive");
  setGlobalValue("HICMA_LRA", "naive");
  timing::start("LR Add Naive");
  AWork += B;
  timing::stopAndPrint("LR Add Naive", 2);
  print("Rel. L2 Error", l2_error(D+D, AWork), false);

  timing::start("Init matrix");
  AWork = A;
  timing::stop("Init matrix");
  print("LR Add Orthogonal");
  setGlobalValue("HICMA_LRA", "rounded_orth");
  timing::start("LR Add Orthogonal");
  AWork += B;
  timing::stopAndPrint("LR Add Orthogonal", 2);
  print("Rel. L2 Error", l2_error(D+D, AWork), false);

  print("-");
  timing::printTime("Init matrix");
  return 0;
}
