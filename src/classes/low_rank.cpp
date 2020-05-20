#include "hicma/classes/low_rank.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/operations/randomized_factorizations.h"
#include "hicma/operations/misc.h"

#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <tuple>


namespace hicma
{

Dense& LowRank::U() { return _U; }
const Dense& LowRank::U() const { return _U; }

Dense& LowRank::S() { return _S; }
const Dense& LowRank::S() const { return _S; }

Dense& LowRank::V() { return _V; }
const Dense& LowRank::V() const { return _V; }

LowRank::LowRank(int64_t n_rows, int64_t n_cols, int64_t k)
: dim{n_rows, n_cols}, rank(k)
{
  U() = Dense(dim[0], k);
  S() = Dense(k, k);
  V() = Dense(k, dim[1]);
}

LowRank::LowRank(const Dense& A, int64_t k)
: Matrix(A), dim{A.dim[0], A.dim[1]} {
  // Rank with oversampling limited by dimensions
  rank = std::min(std::min(k+5, dim[0]), dim[1]);
  std::tie(U(), S(), V()) = rsvd(A, rank);
  U().resize(dim[0], k);
  S().resize(k, k);
  V().resize(k, dim[1]);
  rank = k;
}

void LowRank::mergeU(const LowRank& A, const LowRank& B) {
  assert(rank == A.rank + B.rank);
  for (int64_t i=0; i<dim[0]; i++) {
    for (int64_t j=0; j<A.rank; j++) {
      U()(i,j) = A.U()(i,j);
    }
    for (int64_t j=0; j<B.rank; j++) {
      U()(i,j+A.rank) = B.U()(i,j);
    }
  }
}

void LowRank::mergeS(const LowRank& A, const LowRank& B) {
  assert(rank == A.rank + B.rank);
  for (int64_t i=0; i<A.rank; i++) {
    for (int64_t j=0; j<A.rank; j++) {
      S()(i,j) = A.S()(i,j);
    }
    for (int64_t j=0; j<B.rank; j++) {
      S()(i,j+A.rank) = 0;
    }
  }
  for (int64_t i=0; i<B.rank; i++) {
    for (int64_t j=0; j<A.rank; j++) {
      S()(i+A.rank,j) = 0;
    }
    for (int64_t j=0; j<B.rank; j++) {
      S()(i+A.rank,j+A.rank) = B.S()(i,j);
    }
  }
}

void LowRank::mergeV(const LowRank& A, const LowRank& B) {
  assert(rank == A.rank + B.rank);
  for (int64_t i=0; i<A.rank; i++) {
    for (int64_t j=0; j<dim[1]; j++) {
      V()(i,j) = A.V()(i,j);
    }
  }
  for (int64_t i=0; i<B.rank; i++) {
    for (int64_t j=0; j<dim[1]; j++) {
      V()(i+A.rank,j) = B.V()(i,j);
    }
  }
}

LowRank::LowRank(
  const Dense& U, const Dense& S, const Dense& V
) : _U(U.dim[0], U.dim[1], 0, 0, U),
    _S(S.dim[0], S.dim[1], 0, 0, S),
    _V(V.dim[0], V.dim[1], 0, 0, V),
    dim{U.dim[0], V.dim[1]}, rank(S.dim[0])
{}

LowRank::LowRank(
  int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
  const LowRank& A
) : dim{n_rows, n_cols}, rank(A.rank) {
  assert(row_start+n_rows <= A.dim[0]);
  assert(col_start+n_cols <= A.dim[1]);
  U() = Dense(n_rows, A.rank, row_start, 0, A.U());
  S() = Dense(A.rank, A.rank, 0, 0, A.S());
  V() = Dense(A.rank, n_cols, 0, col_start, A.V());
}

LowRank LowRank::get_part(
  int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start
) const {
  assert(row_start+n_rows <= dim[0]);
  assert(col_start+n_cols <= dim[1]);
  LowRank A(n_rows, n_cols, rank);
  for (int64_t i=0; i<A.U().dim[0]; i++) {
    for (int64_t k=0; k<A.U().dim[1]; k++) {
      A.U()(i, k) = U()(i+row_start, k);
    }
  }
  A.S() = S();
  for (int64_t k=0; k<A.V().dim[0]; k++) {
    for (int64_t j=0; j<A.V().dim[1]; j++) {
      A.V()(k, j) = V()(k, j+col_start);
    }
  }
  return A;
}

} // namespace hicma
