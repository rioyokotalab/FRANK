#include "hicma/classes/low_rank.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/index_range.h"
#include "hicma/classes/node.h"
#include "hicma/operations/randomized_factorizations.h"
#include "hicma/operations/misc/get_dim.h"

#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <tuple>
#include <utility>


namespace hicma
{

std::unique_ptr<Node> LowRank::clone() const {
  return std::make_unique<LowRank>(*this);
}

std::unique_ptr<Node> LowRank::move_clone() {
  return std::make_unique<LowRank>(std::move(*this));
}

const char* LowRank::type() const { return "LowRank"; }

Dense& LowRank::U() { return _U; }
const Dense& LowRank::U() const { return _U; }

Dense& LowRank::S() { return _S; }
const Dense& LowRank::S() const { return _S; }

Dense& LowRank::V() { return _V; }
const Dense& LowRank::V() const { return _V; }

LowRank::LowRank(int m, int n, int k) : dim{m, n}, rank(k) {
  U() = Dense(dim[0], k);
  S() = Dense(k, k);
  V() = Dense(k, dim[1]);
}

LowRank::LowRank(const Dense& A, int k) : Node(A), dim{A.dim[0], A.dim[1]} {
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
  for (int i=0; i<dim[0]; i++) {
    for (int j=0; j<A.rank; j++) {
      U()(i,j) = A.U()(i,j);
    }
    for (int j=0; j<B.rank; j++) {
      U()(i,j+A.rank) = B.U()(i,j);
    }
  }
}

void LowRank::mergeS(const LowRank& A, const LowRank& B) {
  assert(rank == A.rank + B.rank);
  for (int i=0; i<A.rank; i++) {
    for (int j=0; j<A.rank; j++) {
      S()(i,j) = A.S()(i,j);
    }
    for (int j=0; j<B.rank; j++) {
      S()(i,j+A.rank) = 0;
    }
  }
  for (int i=0; i<B.rank; i++) {
    for (int j=0; j<A.rank; j++) {
      S()(i+A.rank,j) = 0;
    }
    for (int j=0; j<B.rank; j++) {
      S()(i+A.rank,j+A.rank) = B.S()(i,j);
    }
  }
}

void LowRank::mergeV(const LowRank& A, const LowRank& B) {
  assert(rank == A.rank + B.rank);
  for (int i=0; i<A.rank; i++) {
    for (int j=0; j<dim[1]; j++) {
      V()(i,j) = A.V()(i,j);
    }
  }
  for (int i=0; i<B.rank; i++) {
    for (int j=0; j<dim[1]; j++) {
      V()(i+A.rank,j) = B.V()(i,j);
    }
  }
}

LowRank LowRank::get_part(
  const IndexRange& row_range, const IndexRange& col_range
) const {
  assert(row_range.start+row_range.length <= dim[0]);
  assert(col_range.start+col_range.length <= dim[1]);
  LowRank A(row_range.length, col_range.length, rank);
  for (int i=0; i<A.U().dim[0]; i++) {
    for (int k=0; k<A.U().dim[1]; k++) {
      A.U()(i, k) = U()(i+row_range.start, k);
    }
  }
  A.S() = S();
  for (int k=0; k<A.V().dim[0]; k++) {
    for (int j=0; j<A.V().dim[1]; j++) {
      A.V()(k, j) = V()(k, j+col_range.start);
    }
  }
  return A;
}

LowRank::LowRank(
  const IndexRange& row_range, const IndexRange& col_range, const LowRank& A
) : dim{row_range.length, col_range.length}, rank(A.rank) {
  assert(row_range.start+row_range.length <= A.dim[0]);
  assert(col_range.start+col_range.length <= A.dim[1]);
  U() = Dense(
    IndexRange(row_range.start, row_range.length), IndexRange(0, A.rank),
    A.U()
  );
  S() = Dense(IndexRange(0, A.rank), IndexRange(0, A.rank), A.S());
  V() = Dense(
    IndexRange(0, A.rank), IndexRange(col_range.start, col_range.length),
    A.V()
  );
}

LowRank::LowRank(
  const Dense& U, const Dense& S, const Dense& V
) : _U(IndexRange(0, U.dim[0]), IndexRange(0, U.dim[1]), U),
    _S(IndexRange(0, S.dim[0]), IndexRange(0, S.dim[1]), S),
    _V(IndexRange(0, V.dim[0]), IndexRange(0, V.dim[1]), V),
    dim{U.dim[0], V.dim[1]}, rank(S.dim[0])
{
}

} // namespace hicma
