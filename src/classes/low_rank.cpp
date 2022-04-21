#include "FRANK/classes/low_rank.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/classes/matrix_proxy.h"
#include "FRANK/operations/LAPACK.h"
#include "FRANK/operations/randomized_factorizations.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"
#include "FRANK/functions.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <tuple>
#include <utility>


namespace FRANK
{

declare_method(LowRank&&, move_from_low_rank, (virtual_<Matrix&>))

LowRank::LowRank(MatrixProxy&& A)
: LowRank(move_from_low_rank(A)) {}

define_method(LowRank&&, move_from_low_rank, (LowRank& A)) {
  return std::move(A);
}

define_method(LowRank&&, move_from_low_rank, (Matrix& A)) {
  omm_error_handler("move_from_low_rank", {A}, __FILE__, __LINE__);
  std::abort();
}

LowRank::LowRank(const Dense& A, const int64_t rank)
: Matrix(A), dim{A.dim[0], A.dim[1]}, rank(rank) {
  // Rank with oversampling limited by dimensions
  std::tie(U, S, V) = rsvd(A, std::min(std::min(rank+5, dim[0]), dim[1]));
  // Reduce to actual desired rank
  U = resize(U, dim[0], rank);
  V = resize(V, rank, dim[1]);
  S = resize(S, rank, rank);
}

LowRank::LowRank(const Dense& A, const double eps)
: Matrix(A), dim{A.dim[0], A.dim[1]}, eps(eps) {
  Dense R;
  std::tie(U, R) = truncated_geqp3(A, eps);
  rank = U.dim[1];
  // Orthogonalize R from RRQR
  S = Dense(rank, rank);
  V = Dense(rank, dim[1]);
  rq(R, S, V);
}

LowRank::LowRank(const Matrix& U, const Dense& S, const Matrix& V, const bool copy)
: dim{get_n_rows(U), get_n_cols(V)}, rank(S.dim[0]),
  U(copy ? MatrixProxy(U) : shallow_copy(U)),
  S(copy ? MatrixProxy(S) : shallow_copy(S)),
  V(copy ? MatrixProxy(V) : shallow_copy(V)) {}

LowRank::LowRank(Dense&& U, Dense&& S, Dense&& V)
: dim{U.dim[0], V.dim[1]}, rank(S.dim[0]),
  U(std::move(U)), S(std::move(S)), V(std::move(V)) {}

} // namespace FRANK
