#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

#include "yorel/yomm2/cute.hpp"


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
  return {std::move(R11), std::move(R22)};
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

std::vector<int> id(Node& A, Node& B, int k) {
  return id_omm(A, B, k);
}

define_method(std::vector<int>, id_omm, (Dense& A, Dense& B, int k)) {
  assert(k <= std::min(A.dim[0], A.dim[1]));
  Dense R(A.dim[1], A.dim[1]);
  std::vector<int> P = geqp3(A, R);
  // First case applies also when A.dim[1] > A.dim[0] end k == A.dim[0]
  if (k < std::min(A.dim[0], A.dim[1]) || A.dim[1] > A.dim[0]) {
    Dense R11, T;
    // TODO Find more abstract way for this. NoCopySplit with designed subnodes?
    std::tie(R11, T) = get_R11_R12(R, k);
    trsm(R11, T, 'u');
    B = interleave_id(T, P);
  } else {
    std::vector<double> x;
    B = interleave_id(Dense(identity, x, k, k), P);
  }
  P.resize(k);
  // Returns the selected columns of A
  return P;
}

// Fallback default, abort with error message
define_method(
  std::vector<int>, id_omm,
  (Node& A, Node& B, [[maybe_unused]] int k)
) {
  std::cerr << "id(";
  std::cerr << A.type() << "," << B.type();
  std::cerr << ") undefined." << std::endl;
  abort();
}


std::tuple<Dense, Dense, Dense> two_sided_id(Node& A, int k) {
  return two_sided_id_omm(A, k);
}

Dense get_cols(const Dense& A, std::vector<int> Pr) {
  Dense B(A.dim[0], Pr.size());
  for (int j=0; j<B.dim[1]; ++j) {
    for (int i=0; i<A.dim[0]; ++i) {
      B(i, j) = A(i, Pr[j]);
    }
  }
  return B;
}

// Fallback default, abort with error message
define_method(DenseTriplet, two_sided_id_omm, (Dense& A, int k)) {
  Dense V(k, A.dim[1]);
  Dense Awork(A);
  std::vector<int> selected_cols = id(Awork, V, k);
  Dense AC = get_cols(A, selected_cols);
  Dense U(k, A.dim[0]);
  AC.transpose();
  Dense ACwork(AC);
  selected_cols = id(ACwork, U, k);
  A = get_cols(AC, selected_cols);
  U.transpose();
  A.transpose();
  return {std::move(U), std::move(A), std::move(V)};
}

// Fallback default, abort with error message
define_method(
  DenseTriplet, two_sided_id_omm,
  (Node& A, [[maybe_unused]] int k)
) {
  std::cerr << "id(";
  std::cerr << A.type();
  std::cerr << ") undefined." << std::endl;
  abort();
}

} // namespace hicma
