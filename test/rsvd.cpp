#include "hicma/hicma.h"

#include "yorel/yomm2/cute.hpp"

#include <cstdint>
#include <tuple>
#include <vector>


using namespace hicma;

int main() {
  yorel::yomm2::update_methods();
  int64_t N = 2048;
  int64_t rank = 16;

  timing::start("Init matrix");
  std::vector<std::vector<double>> randx{get_sorted_random_vector(2*N)};
  Dense D(laplacend, randx, N, N, 0, N);
  timing::stopAndPrint("Init matrix");

  print("RSVD");
  timing::start("Randomized SVD");
  LowRank LR(D, rank);
  timing::stopAndPrint("Randomized SVD", 2);
  print("Rel. L2 Error", l2_error(D, LR), false);

  print("ID");
  Dense U, S, V;
  timing::start("ID");
  Dense Dwork(D);
  std::tie(U, S, V) = id(Dwork, rank);
  timing::stopAndPrint("ID", 2);
  Dense test = gemm(gemm(U, S), V);
  print("Rel. L2 Error", l2_error(D, test), false);

  print("RID");
  timing::start("Randomized ID");
  std::tie(U, S, V) = rid(D, rank+5, rank);
  timing::stopAndPrint("Randomized ID", 2);
  test = gemm(gemm(U, S), V);
  print("Rel. L2 Error", l2_error(D, test), false);
  return 0;
}
