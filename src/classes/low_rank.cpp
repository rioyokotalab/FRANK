#include "hicma/classes/low_rank.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/node.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"
#include "hicma/operations/randomized_factorizations.h"
#include "hicma/operations/misc/get_dim.h"

#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
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

LowRank::LowRank(int64_t m, int64_t n, int64_t k) : dim{m, n}, rank(k) {
  U() = Dense(dim[0], k);
  S() = Dense(k, k);
  V() = Dense(k, dim[1]);
}

LowRank::LowRank(const Dense& A, int64_t k) : Node(A), dim{A.dim[0], A.dim[1]} {
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
  const ClusterTree& node,
  void (*func)(
    Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
  ),
  std::vector<double>& x,
  int64_t k
) : LowRank(Dense(node, func, x), k) {}

LowRank LowRank::get_part(const ClusterTree& node) const {
  assert(node.start[0]+node.dim[0] <= dim[0]);
  assert(node.start[1]+node.dim[1] <= dim[1]);
  LowRank A(node.dim[0], node.dim[1], rank);
  for (int64_t i=0; i<A.U().dim[0]; i++) {
    for (int64_t k=0; k<A.U().dim[1]; k++) {
      A.U()(i, k) = U()(i+node.start[0], k);
    }
  }
  A.S() = S();
  for (int64_t k=0; k<A.V().dim[0]; k++) {
    for (int64_t j=0; j<A.V().dim[1]; j++) {
      A.V()(k, j) = V()(k, j+node.start[1]);
    }
  }
  return A;
}

LowRank::LowRank(const ClusterTree& node, const LowRank& A)
: dim(node.dim), rank(A.rank)
{
  assert(node.start[0]+node.dim[0] <= A.dim[0]);
  assert(node.start[1]+node.dim[1] <= A.dim[1]);
  U() = Dense(ClusterTree(node.dim[0], A.rank, node.start[0], 0),A.U());
  S() = Dense(ClusterTree(A.rank, A.rank), A.S());
  V() = Dense(ClusterTree(A.rank, node.dim[1], 0, node.start[1]), A.V());
}

LowRank::LowRank(
  const Dense& U, const Dense& S, const Dense& V
) : _U(ClusterTree(U.dim[0], U.dim[1]), U),
    _S(ClusterTree(S.dim[0], S.dim[1]), S),
    _V(ClusterTree(V.dim[0], V.dim[1]), V),
    dim{U.dim[0], V.dim[1]}, rank(S.dim[0])
{
}

} // namespace hicma
