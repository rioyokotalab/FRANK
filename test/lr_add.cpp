#include "hicma/hicma.h"

#include "yorel/yomm2/cute.hpp"

#include <cstdint>
#include <vector>


using namespace hicma;

int main() {
  yorel::yomm2::update_methods();
  int64_t N = 2048;
  int64_t rank = 128;
  std::vector<std::vector<double>> randx{get_sorted_random_vector(2*N)};
  print("Time");
  timing::start("Init matrix");
  Dense D(laplacend, randx, N, N, 0, N);
  LowRank A(D, rank);
  LowRank B(D, rank);
  timing::stopAndPrint("Init matrix");
  timing::start("LR Add");
  A += B;
  timing::stopAndPrint("LR Add", 2);
  print("Accuracy");
  print("Rel. L2 Error", l2_error(D+D, A), false);
  return 0;
}
