#include "hicma/hicma.h"

#include "gtest/gtest.h"
#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>


using namespace hicma;

TEST(IDTest, Precision) {
  yorel::yomm2::update_methods();
  // Check whether the Dense(Hierarchical) constructor works correctly.
  int M = 4096;
  int N = 512;
  int k = 32;

  timing::start("Initialization");
  std::vector<double> randx = get_sorted_random_vector(std::max(2*M, 2*N));
  Hierarchical H(laplace1d, randx, M*2, N*2, k, std::max(M, N), 1);
  Dense A(H(M>=N, M<N));
  Dense Awork(A);
  Dense V(k, N);
  timing::stopAndPrint("Initialization");

  timing::start("One-sided ID");
  std::vector<int> Pr;
  std::tie(V, Pr) = one_sided_id(Awork, k);
  Dense Acols = get_cols(A, Pr);
  timing::stopAndPrint("One-sided ID", 1);


  timing::start("Verification");
  Dense Atest(M, N);
  gemm(Acols, V, Atest, 1, 0);
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, Atest), false);
  timing::stopAndPrint("Verification");

  Awork = A;
  timing::start("ID");
  Dense U, S;
  std::tie(U, S, V) = id(Awork, k);
  timing::stopAndPrint("ID", 1);

  timing::start("Verification");
  Dense US(M, k);
  gemm(U, S, US, 1, 0);
  gemm(US, V, Atest, 1, 0);
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, Atest), false);
  timing::stopAndPrint("Verification");
}
