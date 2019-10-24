#include "hicma/operations/id.h"

#include "hicma/node.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations/gemm.h"
#include "hicma/operations/qr.h"
#include "hicma/operations/trsm.h"

#include <algorithm>
#include <cassert>
#include <iostream>
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

Hierarchical split(const Dense& A, int split_row, int split_col) {
  Hierarchical H(2, 2);
  Dense top_left(split_row, split_col);
  for (int i=0; i<top_left.dim[0]; ++i) {
    for (int j = 0; j < top_left.dim[1]; ++j) {
      top_left(i, j) = A(i, j);
    }
  }
  H(0, 0) = top_left;
  Dense top_right(split_row, A.dim[1]-split_col);
  for (int i=0; i<top_right.dim[0]; ++i) {
    for (int j = 0; j < top_right.dim[1]; ++j) {
      top_right(i, j) = A(i, split_col+j);
    }
  }
  H(0, 1) = top_right;
  Dense bottom_left(A.dim[0]-split_row, split_col);
  for (int i=0; i<bottom_left.dim[0]; ++i) {
    for (int j = 0; j < bottom_left.dim[1]; ++j) {
      bottom_left(i, j) = A(split_row+i, j);
    }
  }
  H(1, 0) = bottom_left;
  Dense bottom_right(A.dim[0]-split_row, A.dim[1]-split_col);
  for (int i=0; i<bottom_right.dim[0]; ++i) {
    for (int j = 0; j < bottom_right.dim[1]; ++j) {
      bottom_right(i, j) = A(split_row+i, split_col+j);
    }
  }
  H(1, 1) = bottom_right;
  return H;
}

std::vector<int> qrp(Dense& A, Dense& Q, Dense& R) {
  // Pivoted QR
  for (int i=0; i<std::min(A.dim[0], A.dim[1]); i++) Q(i, i) = 1.0;
  std::vector<int> jpvt(A.dim[1], 0);
  std::vector<double> tau(std::min(A.dim[0], A.dim[1]));
  LAPACKE_dgeqp3(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A[0], A.dim[1],
    &jpvt[0], &tau[0]
  );
  LAPACKE_dormqr(
    LAPACK_ROW_MAJOR,
    'L', 'N',
    A.dim[0], A.dim[1], A.dim[0],
    &A[0], A.dim[0],
    &tau[0],
    &Q[0], Q.dim[1]
  );
  // jpvt is 1-based, bad for indexing!
  for (int& i : jpvt) --i;
  for(int i=0; i<std::min(A.dim[0], A.dim[1]); i++) {
    for(int j=0; j<A.dim[1]; j++) {
      if (j >= i) R(i, j) = A(i, j);
    }
  }
  return jpvt;
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
  Dense Atest(A);
  Dense Q(A.dim[0], A.dim[1]);
  Dense R(A.dim[1], A.dim[1]);
  std::vector<int> P = qrp(A, Q, R);
  if (k < std::max(A.dim[0], A.dim[1])) {
    // The split is inefficient! Full copy, only part needed. Move?
    Hierarchical RH = split(R, k, k);
    Dense& T = static_cast<Dense&>(*RH(0, 1).ptr);
    Dense& R11 = static_cast<Dense&>(*RH(0, 0).ptr);
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
