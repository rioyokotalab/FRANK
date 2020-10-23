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
  int64_t admis = argc > 5 ? atoi(argv[5]) : 0;
  int64_t basis = argc > 6 ? atoi(argv[6]) : 0;
  assert(basis == NORMAL_BASIS || basis == SHARED_BASIS);

  std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};
  Dense x(random_uniform, std::vector<std::vector<double>>(), N);
  Dense b(N);
  Dense D(laplacend, randx, N, N);
  timing::start("Hierarchical compression");
  start_schedule();
  Hierarchical A(
    laplacend, randx, N, N, rank, nleaf, admis, nblocks, nblocks, basis);
  execute_schedule();
  timing::stop("Hierarchical compression");
  // printXML(A);
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
  start_schedule();
  std::tie(L, U) = getrf(A);
  execute_schedule();
  timing::stopAndPrint("LU decomposition", 2);

  timing::start("Solution");
  trsm(L, b, TRSM_LOWER);
  trsm(U, b, TRSM_UPPER);
  timing::stopAndPrint("Solution");

  print("LU Accuracy");
  print("Rel. L2 Error", l2_error(x, b), false);

  print("Overall runtime");
  timing::stopAndPrint("Overall");
  return 0;
}
