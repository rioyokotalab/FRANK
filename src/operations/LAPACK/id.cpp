#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/node.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

std::tuple<Dense, Dense> get_R11_R12(const Dense& R, int64_t k) {
  Dense R11(k, k);
  for (int64_t i=0; i<R11.dim[0]; ++i) {
    for (int64_t j = 0; j < R11.dim[1]; ++j) {
      R11(i, j) = R(i, j);
    }
  }
  Dense R22(k, R.dim[1]-k);
  for (int64_t i=0; i<R22.dim[0]; ++i) {
    for (int64_t j = 0; j < R22.dim[1]; ++j) {
      R22(i, j) = R(i, k+j);
    }
  }
  return {std::move(R11), std::move(R22)};
}

Dense interleave_id(const Dense& A, std::vector<int64_t>& P) {
  int64_t k = P.size() - A.dim[1];
  assert(k >= 0); // 0 case if for k=min(M, N), ie full rank
  Dense Anew(A.dim[0], P.size());
  for (int64_t i=0; i<Anew.dim[0]; ++i) {
    for (int64_t j=0; j<Anew.dim[1]; ++j) {
      Anew(i, P[j]) = j < k ? (i == j ? 1 : 0) : A(i, j-k);
    }
  }
  return Anew;
}

std::tuple<Dense, std::vector<int64_t>> one_sided_id(Node& A, int64_t k) {
  return one_sided_id_omm(A, k);
}

define_method(DenseIndexSetPair, one_sided_id_omm, (Dense& A, int64_t k)) {
  assert(k <= std::min(A.dim[0], A.dim[1]));
  Dense R;
  std::vector<int64_t> selected_cols;
  std::tie(R, selected_cols) = geqp3(A);
  // TODO Consider row index range issues
  Dense col_basis;
  // First case applies also when A.dim[1] > A.dim[0] end k == A.dim[0]
  if (k < std::min(A.dim[0], A.dim[1]) || A.dim[1] > A.dim[0]) {
    Dense R11, T;
    // TODO Find more abstract way for this. NoCopySplit with designed subnodes?
    std::tie(R11, T) = get_R11_R12(R, k);
    trsm(R11, T, TRSM_UPPER);
    col_basis = interleave_id(T, selected_cols);
  } else {
    std::vector<double> x;
    col_basis = interleave_id(Dense(identity, x, k, k), selected_cols);
  }
  selected_cols.resize(k);
  // Returns the selected columns of A
  return {std::move(col_basis), std::move(selected_cols)};
}

// Fallback default, abort with error message
define_method(
  DenseIndexSetPair, one_sided_id_omm,
  (Node& A, [[maybe_unused]] int64_t k)
) {
  omm_error_handler("id", {A}, __FILE__, __LINE__);
  abort();
}


std::tuple<Dense, Dense, Dense> id(Node& A, int64_t k) { return id_omm(A, k); }

Dense get_cols(const Dense& A, std::vector<int64_t> Pr) {
  Dense B(A.dim[0], Pr.size());
  for (int64_t j=0; j<B.dim[1]; ++j) {
    for (int64_t i=0; i<A.dim[0]; ++i) {
      B(i, j) = A(i, Pr[j]);
    }
  }
  return B;
}

// Fallback default, abort with error message
define_method(DenseTriplet, id_omm, (Dense& A, int64_t k)) {
  Dense V(k, A.dim[1]);
  Dense Awork(A);
  std::vector<int64_t> selected_cols;
  std::tie(V, selected_cols) = one_sided_id(Awork, k);
  Dense AC = get_cols(A, selected_cols);
  Dense U(k, A.dim[0]);
  AC.transpose();
  Dense ACwork(AC);
  std::tie(U, selected_cols) = one_sided_id(ACwork, k);
  A = get_cols(AC, selected_cols);
  U.transpose();
  A.transpose();
  return {std::move(U), std::move(A), std::move(V)};
}

// Fallback default, abort with error message
define_method(
  DenseTriplet, id_omm,
  (Node& A, [[maybe_unused]] int64_t k)
) {
  omm_error_handler("id", {A}, __FILE__, __LINE__);
  abort();
}

} // namespace hicma
