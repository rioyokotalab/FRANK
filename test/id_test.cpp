#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/functions.h"
#include "hicma/hierarchical.h"
#include "hicma/operations/gemm.h"
#include "hicma/operations/id.h"
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

Dense get_rows(const Dense& A, std::vector<int>& Pl) {
  Dense B(A.dim[0], Pl.size());
  for (int i=0; i<Pl.size(); ++i) {
    for (int j=0; j<A.dim[1]; ++j) {
      B(i, j) = A(Pl[i], j);
    }
  }
  return B;
}

TEST(IDTest, Precision) {
  yorel::multi_methods::initialize();
  // Check whether the Dense(Hierarchical) constructor works correctly.
  int M = 32;
  int N = 128;
  int k = 3;
  std::vector<double> randx(std::max(M, N)*2);
  for (double& a : randx) a = drand48();
  std::sort(randx.begin(), randx.end());
  Hierarchical H(laplace1d, randx, M*2, N*2, k, std::max(M, N), 1);
  Dense& A = static_cast<Dense&>(*H(M>=N, M<N).ptr);
  Dense Awork(A);

  Dense V(k, N);
  std::vector<int> Pr = id(Awork, V, k);

  Dense Acols = get_cols(A, Pr);
  Dense Atest(M, N);
  gemm(Acols, V, Atest, 1, 0);

  start("Verification");
  double norm = A.norm();
  double diff = (A - Atest).norm();
  stop("Verification");
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
}
