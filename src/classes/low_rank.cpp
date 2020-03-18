#include "hicma/classes/low_rank.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/operations/randomized/rsvd.h"

#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <tuple>
#include <utility>

namespace hicma {

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

  LowRank::LowRank(const Node& node, int k, bool node_only)
  : Node(node), dim{row_range.length, col_range.length}, rank(k) {
    if (!node_only) {
      U() = Dense(dim[0], k, i_abs, j_abs, level);
      S() = Dense(k, k, i_abs, j_abs, level);
      V() = Dense(k, dim[1], i_abs, j_abs, level);
    }
  }

  LowRank::LowRank(
    int m, int n,
    int k,
    int i_abs, int j_abs,
    int level
  ) : LowRank(
    Node(i_abs, j_abs, level, IndexRange(0, m), IndexRange(0, n)),
    k
  ) {}

  LowRank::LowRank(const Dense& A, int k)
  : Node(A), dim{A.dim[0], A.dim[1]} {
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

  LowRank LowRank::get_part(const Node& node) const {
    assert(is_child(node));
    LowRank A(node, rank);
    int rel_row_begin = A.row_range.start - row_range.start;
    int rel_col_begin = A.col_range.start - col_range.start;
    for (int i=0; i<A.U().dim[0]; i++) {
      for (int k=0; k<A.U().dim[1]; k++) {
        A.U()(i, k) = U()(i+rel_row_begin, k);
      }
    }
    A.S() = S();
    for (int k=0; k<A.V().dim[0]; k++) {
      for (int j=0; j<A.V().dim[1]; j++) {
        A.V()(k, j) = V()(k, j+rel_col_begin);
      }
    }
    return A;
  }

} // namespace hicma
