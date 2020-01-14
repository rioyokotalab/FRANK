#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <utility>

#include "gtest/gtest.h"
#include "yorel/multi_methods.hpp"

using namespace hicma;

TEST(IDTest, Precision) {
  yorel::multi_methods::initialize();
  // Check whether the Dense(Hierarchical) constructor works correctly.
  int M = 4096;
  int N = 512;
  int k = 32;

  timing::start("Initialization");
  std::vector<double> randx(std::max(M, N)*2);
  for (double& a : randx) a = drand48();
  std::sort(randx.begin(), randx.end());
  Hierarchical H(laplace1d, randx, M*2, N*2, k, std::max(M, N), 1);
  Dense A(H(M>=N, M<N));
  Dense Awork(A);
  Dense V(k, N);
  timing::stopAndPrint("Initialization");

  timing::start("ID");
  std::vector<int> Pr = id(Awork, V, k);
  Dense Acols = get_cols(A, Pr);
  timing::stopAndPrint("ID", 1);


  timing::start("Verification");
  Dense Atest(M, N);
  gemm(Acols, V, Atest, 1, 0);
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, Atest), false);
  timing::stopAndPrint("Verification");

  Awork = A;
  timing::start("Two-sided ID");
  Dense U, S;
  std::tie(U, S, V) = two_sided_id(Awork, k);
  timing::stopAndPrint("Two-sided ID", 1);

  timing::start("Verification");
  Dense US(M, k);
  gemm(U, S, US, 1, 0);
  gemm(US, V, Atest, 1, 0);
  print("Compression Accuracy");
  print("Rel. L2 Error", l2_error(A, Atest), false);
  timing::stopAndPrint("Verification");
}
