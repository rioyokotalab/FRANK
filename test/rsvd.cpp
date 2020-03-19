#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

#include "yorel/yomm2/cute.hpp"

using namespace hicma;

int main() {
  yorel::yomm2::update_methods();
  int N = 2048;
  int rank = 16;

  timing::start("Init matrix");
  std::vector<double> randx = get_sorted_random_vector(2*N);
  Dense D(laplace1d, randx, N, N, 0, N);
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
  Dense US(U.dim[0], S.dim[1]);
  Dense test(U.dim[0], V.dim[1]);
  gemm(U, S, US, 1, 0);
  gemm(US, V, test, 1, 0);
  print("Rel. L2 Error", l2_error(D, test), false);

  print("RID");
  timing::start("Randomized ID");
  std::tie(U, S, V) = rid(D, rank+5, rank);
  timing::stopAndPrint("Randomized ID", 2);
  gemm(U, S, US, 1, 0);
  gemm(US, V, test, 1, 0);
  print("Rel. L2 Error", l2_error(D, test), false);
  return 0;
}
