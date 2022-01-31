#include "hicma/classes/low_rank.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/extension_headers/classes.h"
#include "hicma/operations/randomized_factorizations.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <tuple>
#include <utility>


namespace hicma
{

//explicit template initialization
//only double matrix is available
template class LowRank<double>;

declare_method(LowRank<double>&&, move_from_low_rank, (virtual_<Matrix&>))

template<typename T>
LowRank<T>::LowRank(MatrixProxy&& A)
: LowRank(move_from_low_rank(A)) {}

define_method(LowRank<double>&&, move_from_low_rank, (LowRank<double>& A)) {
  return std::move(A);
}

define_method(LowRank<double>&&, move_from_low_rank, (Matrix& A)) {
  omm_error_handler("move_from_low_rank", {A}, __FILE__, __LINE__);
  std::abort();
}

template<typename T>
LowRank<T>::LowRank(const Dense<T>& A, int64_t rank)
: Matrix(A), dim{A.dim[0], A.dim[1]}, rank(rank) {
  // Rank with oversampling limited by dimensions
  std::tie(U, S, V) = rsvd(A, std::min(std::min(rank+5, dim[0]), dim[1]));
  // Reduce to actual desired rank
  U = resize(U, dim[0], rank);
  V = resize(V, rank, dim[1]);
  S = resize(S, rank, rank);
}

template<typename T>
LowRank<T>::LowRank(const Matrix& U, const Dense<T>& S, const Matrix& V, bool copy)
: dim{get_n_rows(U), get_n_cols(V)}, rank(S.dim[0]),
  U(copy ? MatrixProxy(U) : shallow_copy(U)),
  S(copy ? MatrixProxy(S) : shallow_copy(S)),
  V(copy ? MatrixProxy(V) : shallow_copy(V)) {}

template<typename T>
LowRank<T>::LowRank(Dense<T>&& U, Dense<T>&& S, Dense<T>&& V)
: dim{U.dim[0], V.dim[1]}, rank(S.dim[0]),
  U(std::move(U)), S(std::move(S)), V(std::move(V)) {}

} // namespace hicma
