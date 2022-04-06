#include "hicma/hicma.h"

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>


using namespace hicma;

int main(int argc, char** argv) {
  timing::start("Overall");
  hicma::initialize();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t nleaf = argc > 2 ? atoi(argv[2]) : 16;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 8;
  int64_t nblocks = argc > 4 ? atoi(argv[4]) : 2;
  double admis = argc > 5 ? atof(argv[5]) : 0;

  std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};
  Dense x(random_uniform, {}, N);
  Dense b(N);
  Dense D(laplacend, randx, N, N);
  timing::start("Hierarchical compression");
  Hierarchical A(laplacend, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  timing::stop("Hierarchical compression");
  // write_JSON(A);
  gemm(A, x, b, 1, 1);

  print("Compression Accuracy");
  timing::start("Compression accuracy check");
  double comp_error = l2_error(A, D);
  double comp_rate = double(get_memory_usage(D)) / double(get_memory_usage(A));
  timing::stop("Compression accuracy check");
  print("Rel. L2 Error", comp_error, false);
  print("Compression factor", comp_rate);
  print("Time");
  timing::printTime("Hierarchical compression");
  timing::printTime("Compression accuracy check");

  timing::start("LU decomposition");
  Hierarchical L, U;
  std::tie(L, U) = getrf(A);
  timing::stopAndPrint("LU decomposition", 2);

  timing::start("Solution");
  trsm(L, b, Mode::Lower);
  trsm(U, b, Mode::Upper);
  timing::stopAndPrint("Solution");

  print("LU Accuracy");
  print("Rel. L2 Error", l2_error(x, b), false);

  print("Overall runtime");
  timing::stopAndPrint("Overall");
  return 0;
}
