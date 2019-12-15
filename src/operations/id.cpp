#include "hicma/operations/id.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations/gemm.h"
#include "hicma/operations/geqp3.h"
#include "hicma/operations/qr.h"
#include "hicma/operations/trsm.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

std::tuple<Dense, Dense> get_R11_R12(const Dense& R, int k) {
  Dense R11(k, k);
  for (int i=0; i<R11.dim[0]; ++i) {
    for (int j = 0; j < R11.dim[1]; ++j) {
      R11(i, j) = R(i, j);
    }
  }
  Dense R22(k, R.dim[1]-k);
  for (int i=0; i<R22.dim[0]; ++i) {
    for (int j = 0; j < R22.dim[1]; ++j) {
      R22(i, j) = R(i, k+j);
    }
  }
  return {R11, R22};
}

Dense interleave_id(const Dense& A, std::vector<int>& P) {
  int k = P.size() - A.dim[1];
  assert(k >= 0); // 0 case if for k=min(M, N), ie full rank
  Dense Anew(A.dim[0], P.size());
  for (int i=0; i<Anew.dim[0]; ++i) {
    for (int j=0; j<Anew.dim[1]; ++j) {
      Anew(i, P[j]) = j < k ? (i == j ? 1 : 0) : A(i, j-k);
    }
  }
  return Anew;
}

std::vector<int> id(Node& A, Node& B, const int k) {
  return id_omm(A, B, k);
}

BEGIN_SPECIALIZATION(
  id_omm, std::vector<int>,
  Dense& A, Dense& B,
  const int k
) {
  assert(k <= std::min(A.dim[0], A.dim[1]));
  Dense R(A.dim[0], A.dim[1]);
  std::vector<int> P = geqp3(A, R);
  // First case applies also when A.dim[1] > A.dim[0] end k == A.dim[0]
  if (k < std::min(A.dim[0], A.dim[1]) || A.dim[1] > A.dim[0]) {
    Dense R11, T;
    std::tie(R11, T) = get_R11_R12(R, k);
    cblas_dtrsm(
      CblasRowMajor,
      CblasLeft, CblasUpper,
      CblasNoTrans, CblasNonUnit,
      T.dim[0], T.dim[1],
      1,
      &R11[0], R11.dim[1],
      &T[0], T.dim[1]
    );
    B = interleave_id(T, P);
  } else {
    std::vector<double> x;
    B = interleave_id(Dense(identity, x, k, k), P);
  }
  P.resize(k);
  // Returns the selected columns of A
  return P;
} END_SPECIALIZATION;

// Fallback default, abort with error message
BEGIN_SPECIALIZATION(
  id_omm, std::vector<int>,
  Node& A, Node& B,
  const int k
) {
  std::cerr << "id(";
  std::cerr << A.type() << "," << B.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
