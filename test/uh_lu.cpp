#include "hicma/hicma.h"

#include "yorel/yomm2/cute.hpp"

#include <tuple>
#include <vector>


using namespace hicma;

int main(int argc, char const *argv[])
{
  yorel::yomm2::update_methods();
  int N = argc > 1 ? atoi(argv[1]) : 256;
  int nblocks = argc > 2 ? atoi(argv[2]) : 2;
  int rank = argc > 3 ? atoi(argv[3]) : 8;
  int admis = 0;
  int nleaf = N/nblocks;
  std::vector<double> randx = get_sorted_random_vector(N);
  timing::start("Init matrix");
  UniformHierarchical A(
    laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  timing::stopAndPrint("Init matrix", 1);
  timing::start("Init matrix SVD");
  UniformHierarchical A_svd(
    laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks, true);
  timing::stopAndPrint("Init matrix SVD", 1);
  // printXML(A);
  Hierarchical H(
    laplace1d, randx, N, N, rank, nleaf, admis, nblocks, nblocks);
  Hierarchical D(laplace1d, randx, N, N, rank, nleaf, N/nblocks, nblocks, nblocks);
  Dense rand(random_normal, randx, N, N);
  Dense x(random_uniform, randx, N);
  Dense b(N);
  gemm(A, x, b, 1, 1);

  timing::start("Verification");
  print("Compression Accuracy");
  print("H Rel. L2 Error", l2_error(D, H), false);
  print("UH Rel. L2 Error", l2_error(D, A), false);
  timing::stopAndPrint("Verification");

  print("GEMM");
  timing::start("GEMM");
  Dense test1(N, N);
  gemm(A, rand, test1, 1, 0);
  timing::stopAndPrint("GEMM", 1);

  print("LU");
  timing::start("UBLR LU");
  UniformHierarchical L, U;
  std::tie(L, U) = getrf(A);
  timing::stopAndPrint("UBLR LU", 2);

  timing::start("Verification");
  trsm(L, b, TRSM_LOWER);
  trsm(U, b, TRSM_UPPER);
  print("UH Rel. L2 Error", l2_error(x, b), false);
  timing::stopAndPrint("Verification");

  return 0;
}
