#include "hicma/hicma.h"

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>
#include <iostream>


using namespace hicma;
using namespace std;

int main(int argc, char** argv) {
  timing::start("Overall");
  hicma::initialize();

  // int64_t nleaf = atoi(argv[1]);
  // int64_t rank = atoi(argv[2]);
  // int64_t admis = atoi(argv[3]);

  int64_t nleaf = 1024;
  int64_t rank = 8;
  int64_t admis = 0;
  
  // int64_t N = 65536;
  int64_t N = 8192;
  int64_t basis = 0;
  int64_t nblocks = N / nleaf;
  assert(basis == NORMAL_BASIS || basis == SHARED_BASIS);

  Dense MAT(N, N);
  std::cout << "reading hicmat file... \n";
  read_hicmat("/groups2/gaa50004/acb10922qh/lorapo-3d-matern-8192-1024.hicmat", MAT);
  // read_hicmat(string("/groups2/gaa50004/acb10922qh/lorapo-3d-matern-8192-") +
  //             string(argv[1]) + ".hicmat", MAT);
  // read_hicmat(string("/groups2/gaa50004/acb10922qh/lorapo-3d-matern-65536-") +
  //             string(argv[1]) + ".hicmat", MAT);
  std::cout << "done reading hicmat... \n";
  
  std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};
  Dense x(random_uniform, std::vector<std::vector<double>>(), N);
  Dense b(N);
  timing::start("Hierarchical compression");
  start_schedule();
  Hierarchical A(MAT, rank, nleaf, admis, nblocks, nblocks, basis);
  execute_schedule();
  timing::stop("Hierarchical compression");
  printXML(A);
  gemm(A, x, b, 1, 1);

  print("Compression Accuracy");
  timing::start("Compression accuracy check");
  double comp_error = l2_error(A, MAT);
  double comp_rate = double(get_memory_usage(MAT)) / double(get_memory_usage(A));
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
