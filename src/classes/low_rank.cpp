#include "hicma/classes/low_rank.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/shared_basis.h"
#include "hicma/extension_headers/classes.h"
#include "hicma/operations/randomized_factorizations.h"
#include "hicma/operations/misc.h"

#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <tuple>


namespace hicma
{

LowRank::LowRank(int64_t n_rows, int64_t n_cols, int64_t k)
: dim{n_rows, n_cols}, rank(k)
{
  U = Dense(dim[0], k);
  S = Dense(k, k);
  V = Dense(k, dim[1]);
}

LowRank::LowRank(const Dense& A, int64_t k, svdType type)
: Matrix(A), dim{A.dim[0], A.dim[1]} {
  // Rank with oversampling limited by dimensions
  rank = std::min(std::min(k+5, dim[0]), dim[1]);
  Dense U_full, V_full;
  switch (type){
    case powIt:         std::tie(U_full, S, V_full) = rsvd_pow(A, rank, 2);
                      break;

    case powOrtho:    std::tie(U_full, S, V_full) = rsvd_powOrtho(A, rank, 2);
                      break;

    case singlePass:  std::tie(U_full, S, V_full) = rsvd_singlePass(A, rank);
                      break;

    default:          std::tie(U_full, S, V_full) = rsvd(A, rank);
  }
  //std::tie(U_full, S, V_full) = rsvd(A, rank);
  U_full.resize(dim[0], k);
  U = std::move(U_full);
  V_full.resize(k, dim[1]);
  V = std::move(V_full);
  S.resize(k, k);
  rank = k;
}

void LowRank::mergeU(const LowRank& A, const LowRank& B) {
  assert(rank == A.rank + B.rank);
  Hierarchical U_new(1, 2);
  U_new[0] = get_part(A.U, A.dim[0], A.rank, 0, 0);
  U_new[1] = get_part(B.U, B.dim[0], B.rank, 0, 0);
  U = Dense(U_new);
}

void LowRank::mergeS(const LowRank& A, const LowRank& B) {
  assert(rank == A.rank + B.rank);
  for (int64_t i=0; i<A.rank; i++) {
    for (int64_t j=0; j<A.rank; j++) {
      S(i,j) = A.S(i,j);
    }
    for (int64_t j=0; j<B.rank; j++) {
      S(i,j+A.rank) = 0;
    }
  }
  for (int64_t i=0; i<B.rank; i++) {
    for (int64_t j=0; j<A.rank; j++) {
      S(i+A.rank,j) = 0;
    }
    for (int64_t j=0; j<B.rank; j++) {
      S(i+A.rank,j+A.rank) = B.S(i,j);
    }
  }
}

void LowRank::mergeV(const LowRank& A, const LowRank& B) {
  assert(rank == A.rank + B.rank);
  Hierarchical V_new(2, 1);
  V_new[0] = get_part(A.V, A.rank, A.dim[1], 0, 0);
  V_new[1] = get_part(B.V, B.rank, B.dim[1], 0, 0);
  V = Dense(V_new);
}

LowRank::LowRank(
  const Matrix& U, const Dense& S, const Matrix& V, bool copy_S
) : U(share_basis(U)), V(share_basis(V)),
    S(S, S.dim[0], S.dim[1], 0, 0, copy_S),
    dim{get_n_rows(U), get_n_cols(V)}, rank(S.dim[0]) {}

LowRank::LowRank(
  const LowRank& A,
  int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
  bool copy
) : dim{n_rows, n_cols}, rank(A.rank) {
  assert(row_start+n_rows <= A.dim[0]);
  assert(col_start+n_cols <= A.dim[1]);
  U = get_part(A.U, n_rows, A.rank, row_start, 0, copy);
  S = Dense(A.S, A.rank, A.rank, 0, 0, copy);
  V = get_part(A.V, A.rank, n_cols, 0, col_start, copy);
}

} // namespace hicma
