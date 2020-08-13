#include "hicma/hicma.h"

#include <cstdint>
#include <tuple>
#include <vector>
#include <cmath>


using namespace hicma;

int main() {
  hicma::initialize();
  int64_t N = 2048;
  int64_t rank = 1024;

  timing::start("Init matrix");
  std::vector<std::vector<double>> randx{get_sorted_random_vector(2*N)};
  Dense D(laplacend, randx, N, N, 0, N);
  timing::stopAndPrint("Init matrix");
  Dense sing = get_singular_values(D);
  double err = 0.0;
  for (int i=rank; i<N; ++i)
    err += sing[i] * sing[i];
  print("Minimal possible error", std::sqrt(err), false);

  print("RSVD");
  timing::start("Randomized SVD");
  LowRank LR(D, rank);
  timing::stopAndPrint("Randomized SVD", 2);
  print("Rel. L2 Error", l2_error(D, LR), false);

  print("RSVD - Power Iteration");
  timing::start("Randomized SVD pow");
  LowRank LRpow(D, rank, powIt);
  timing::stopAndPrint("Randomized SVD pow", 2);
  print("Rel. L2 Error", l2_error(D, LRpow), false);

  print("RSVD - Orthonormalized Power Iteration");
  timing::start("Randomized SVD powOrtho");
  LowRank LRpowOrtho(D, rank, powOrtho);
  timing::stopAndPrint("Randomized SVD powOrtho", 2);
  print("Rel. L2 Error", l2_error(D, LRpowOrtho), false);

  print("RSVD - Single Pass");
  timing::start("Randomized SVD singlePass");
  LowRank LRsinglePass(D, rank, singlePass);
  timing::stopAndPrint("Randomized SVD singlePass", 2);
  print("Rel. L2 Error", l2_error(D, LRsinglePass), false);
  

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
