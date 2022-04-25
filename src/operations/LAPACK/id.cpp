#include "FRANK/operations/LAPACK.h"

#include "FRANK/definitions.h"
#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/classes/initialization_helpers/index_range.h"
#include "FRANK/functions.h"
#include "FRANK/operations/BLAS.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>


namespace FRANK
{

Dense interleave_id(const Dense& A, std::vector<int64_t>& P) {
  const int64_t k = P.size() - A.dim[1];
  assert(k >= 0); // 0 case if for k=min(M, N), ie full rank
  Dense Anew(A.dim[0], P.size());
  for (int64_t i=0; i<Anew.dim[0]; ++i) {
    for (int64_t j=0; j<Anew.dim[1]; ++j) {
      Anew(i, P[j]) = j < k ? (i == j ? 1 : 0) : A(i, j-k);
    }
  }
  return Anew;
}

declare_method(
  DenseIndexSetPair, one_sided_id_omm, (virtual_<Matrix&>, const int64_t)
)

std::tuple<Dense, std::vector<int64_t>> one_sided_id(Matrix& A, const int64_t k) {
  return one_sided_id_omm(A, k);
}

define_method(DenseIndexSetPair, one_sided_id_omm, (Dense& A, const int64_t k)) {
  assert(k <= std::min(A.dim[0], A.dim[1]));
  Dense R;
  std::vector<int64_t> selected_cols;
  std::tie(R, selected_cols) = geqp3(A);
  // TODO Consider row index range issues
  Dense col_basis;
  // First case applies also when A.dim[1] > A.dim[0] end k == A.dim[0]
  if (k < std::min(A.dim[0], A.dim[1]) || A.dim[1] > A.dim[0]) {
    // Get R11 (split[0]) and R22 (split[1])
    std::vector<Dense> split = R.split(
      IndexRange(0, R.dim[0]).split_at(k), IndexRange(0, R.dim[1]).split_at(k)
    );
    trsm(split[0], split[1], Mode::Upper);
    col_basis = interleave_id(split[1], selected_cols);
  } else {
    col_basis = interleave_id(
      Dense(identity, {}, k, k), selected_cols);
  }
  selected_cols.resize(k);
  // Returns the selected columns of A
  return {std::move(col_basis), std::move(selected_cols)};
}

// Fallback default, abort with error message
define_method(DenseIndexSetPair, one_sided_id_omm, (Matrix& A, const int64_t)) {
  omm_error_handler("id", {A}, __FILE__, __LINE__);
  std::abort();
}

declare_method(DenseTriplet, id_omm, (virtual_<Matrix&>, const int64_t))

std::tuple<Dense, Dense, Dense> id(Matrix& A, const int64_t k) {
  return id_omm(A, k);
}

Dense get_cols(const Dense& A, std::vector<int64_t> Pr) {
  Dense B(A.dim[0], Pr.size());
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<B.dim[1]; ++j) {
      B(i, j) = A(i, Pr[j]);
    }
  }
  return B;
}

Dense get_rows(const Dense& A, std::vector<int64_t> Pr) {
  Dense B(Pr.size(), A.dim[1]);
  for (int64_t i=0; i<B.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      B(i, j) = A(Pr[i], j);
    }
  }
  return B;
}

define_method(DenseTriplet, id_omm, (Dense& A, const int64_t k)) {
  Dense V(k, A.dim[1]);
  Dense Awork(A);
  std::vector<int64_t> selected_cols;
  std::tie(V, selected_cols) = one_sided_id(Awork, k);
  Dense AC = get_cols(A, selected_cols);
  Dense ACwork = transpose(AC);
  Dense Ut(k, A.dim[0]);
  std::vector<int64_t> selected_rows;
  std::tie(Ut, selected_rows) = one_sided_id(ACwork, k);
  return {transpose(Ut), get_rows(AC, selected_rows), std::move(V)};
}

// Fallback default, abort with error message
define_method(DenseTriplet, id_omm, (Matrix& A, const int64_t)) {
  omm_error_handler("id", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
