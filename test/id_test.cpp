#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/functions.h"
#include "hicma/hierarchical.h"
#include "hicma/operations/gemm.h"
#include "hicma/operations/id.h"
#include "hicma/operations/norm.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "yorel/multi_methods.hpp"

using namespace hicma;

Dense get_cols(const Dense& A, std::vector<int> Pr) {
  Dense B(A.dim[0], Pr.size());
  for (int j=0; j<Pr.size(); ++j) {
    for (int i=0; i<A.dim[0]; ++i) {
      B(i, j) = A(i, Pr[j]);
    }
  }
  return B;
}

TEST(IDTest, Precision) {
  yorel::multi_methods::initialize();
  // Check whether the Dense(Hierarchical) constructor works correctly.
  int M = 1024;
  int N = 32;
  int k = 6;

  start("Initialization");
  std::vector<double> randx(std::max(M, N)*2);
  for (double& a : randx) a = drand48();
  std::sort(randx.begin(), randx.end());
  Hierarchical H(laplace1d, randx, M*2, N*2, k, std::max(M, N), 1);
  Dense& A = static_cast<Dense&>(*H(M>=N, M<N));
  Dense Awork(A);
  stop("Initialization");

  Dense V(k, N);
  start("ID");
  std::vector<int> Pr = id(Awork, V, k);
  stop("ID");

  Dense Acols = get_cols(A, Pr);
  Dense Atest(M, N);
  gemm(Acols, V, Atest, 1, 0);

  start("Verification");
  double l2 = norm(A);
  double diff = norm(A - Atest);
  stop("Verification");
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/l2), false);
}
