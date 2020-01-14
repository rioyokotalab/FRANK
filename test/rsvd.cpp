#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main(int argc, char** argv) {
  yorel::multi_methods::initialize();
  int N = 2048;
  int rank = 16;

  timing::start("Init matrix");
  std::vector<double> randx(2*N);
  for (int i=0; i<2*N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  Dense D(laplace1d, randx, N, N, 0, N);
  timing::stopAndPrint("Init matrix");

  print("RSVD");
  timing::start("Randomized SVD");
  LowRank LR(D, rank);
  timing::stopAndPrint("Randomized SVD", 2);
  print("Rel. L2 Error", l2_error(D, LR), false);

  print("ID");
  Dense U, S, V;
  timing::start("Two-sided ID");
  Dense Dwork(D);
  std::tie(U, S, V) = two_sided_id(Dwork, rank);
  timing::stopAndPrint("Two-sided ID", 2);
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
