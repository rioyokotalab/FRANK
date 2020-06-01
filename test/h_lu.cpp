#include "hicma/hicma.h"

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>


using namespace hicma;

int main(int argc, char** argv) {
  hicma::initialize();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t nleaf = argc > 2 ? atoi(argv[2]) : 16;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 8;
  int64_t nblocks = argc > 4 ? atoi(argv[4]) : 2;
  int64_t admis = argc > 5 ? atoi(argv[5]) : 0;
  int64_t basis = argc > 6 ? atoi(argv[6]) : 0;
  assert(basis == NORMAL_BASIS || basis == SHARED_BASIS);
  std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};
  timing::start("Init matrix");
  timing::start("CPU compression");
  Hierarchical A(
    laplacend, randx, N, N, rank, nleaf, admis, nblocks, nblocks, basis);
  timing::stop("CPU compression");
  rsvd_batch();
  // printXML(A);
  admis = N / nleaf; // Full rank
  Dense x(random_uniform, std::vector<std::vector<double>>(), N);
  Dense b(N);
  // timing::start("Dense tree");
  // Hierarchical D(laplacend, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  // timing::stop("Dense tree");
  // timing::start("Verification time");
  // print("Compression Accuracy");
  // print("Rel. L2 Error", l2_error(A, D), false);
  // timing::stop("Verification time");
  print("Time");
  gemm(A, x, b, 1, 1);
  gemm_batch();
  timing::stopAndPrint("Init matrix");
  timing::start("LU decomposition");
  Hierarchical L, U;
  std::tie(L, U) = getrf(A);
  timing::stopAndPrint("LU decomposition", 2);
  timing::start("Verification time");
  timing::start("Forward substitution");
  trsm(L, b, TRSM_LOWER);
  timing::stop("Forward substitution");
  timing::start("Backward substitution");
  trsm(U, b, TRSM_UPPER);
  timing::stop("Backward substitution");
  print("LU Accuracy");
  print("Rel. L2 Error", l2_error(x, b), false);
  timing::stopAndPrint("Verification time");
  return 0;
}
