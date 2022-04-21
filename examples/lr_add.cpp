#include "FRANK/FRANK.h"

#include <cstdint>
#include <vector>


using namespace FRANK;

int main(int argc, char** argv) {
  FRANK::initialize();
  const int64_t N = argc > 1 ? atoi(argv[1]) : 128;
  const int64_t rank = argc > 2 ? atoi(argv[2]) : 16;
  const std::vector<std::vector<double>> randx{ get_sorted_random_vector(2*N) };
  timing::start("Init matrix");
  const Dense D(laplacend, randx, N, N, 0, N);
  const LowRank A(D, rank);
  const LowRank B(D, rank);
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
  setGlobalValue("FRANK_LRA", "naive");
  timing::start("LR Add Naive");
  AWork += B;
  timing::stopAndPrint("LR Add Naive", 2);
  print("Rel. L2 Error", l2_error(D+D, AWork), false);

  timing::start("Init matrix");
  AWork = A;
  timing::stop("Init matrix");
  print("LR Add Orthogonal");
  setGlobalValue("FRANK_LRA", "rounded_addition");
  timing::start("LR Add Orthogonal");
  AWork += B;
  timing::stopAndPrint("LR Add Orthogonal", 2);
  print("Rel. L2 Error", l2_error(D+D, AWork), false);

  print("-");
  timing::printTime("Init matrix");
  return 0;
}
