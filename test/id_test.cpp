#include "hicma/hicma.h"

#include "gtest/gtest.h"
#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>


using namespace hicma;

TEST(IDTest, Precision) {
  hicma::initialize();
  // Check whether the Dense(Hierarchical) constructor works correctly.
  int64_t M = 4096;
  int64_t N = 512;
  int64_t k = 32;

  timing::start("Initialization");
  std::vector<std::vector<double>> randx = {
    get_sorted_random_vector(std::max(2*M, 2*N))
  };
  Hierarchical H(laplacend, randx, M*2, N*2, k, std::max(M, N), 1);
  Dense A(H(M>=N, M<N));
  Dense Awork(A);
  Dense V(k, N);
  timing::stopAndPrint("Initialization");

  timing::start("One-sided ID");
  std::vector<int64_t> Pr;
  std::tie(V, Pr) = one_sided_id(Awork, k);
  Dense Acols = get_cols(A, Pr);
  timing::stopAndPrint("One-sided ID", 1);


  timing::start("Verification");
  Dense Atest = gemm(Acols, V);
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, Atest), false);
  timing::stopAndPrint("Verification");

  Awork = A;
  timing::start("ID");
  Dense U, S;
  std::tie(U, S, V) = id(Awork, k);
  timing::stopAndPrint("ID", 1);

  timing::start("Verification");
  Atest = gemm(gemm(U, S), V);
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, Atest), false);
  timing::stopAndPrint("Verification");
}
