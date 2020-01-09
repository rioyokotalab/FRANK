#include "hicma/classes/low_rank.h"

#include "hicma/classes/node.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/LAPACK/qr.h"
#include "hicma/operations/LAPACK/svd.h"
#include "hicma/operations/randomized/rsvd.h"
#include "hicma/util/print.h"
#include "hicma/util/counter.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <tuple>
#include <utility>

#include "yorel/multi_methods.hpp"

namespace hicma {

  LowRank::LowRank() : dim{0, 0}, rank(0) { MM_INIT(); }

  LowRank::~LowRank() = default;

  LowRank::LowRank(const LowRank& A) {
    MM_INIT();
    *this = A;
  }

  LowRank& LowRank::operator=(const LowRank& A) = default;

  LowRank::LowRank(LowRank&& A) {
    MM_INIT();
    *this = std::move(A);
  }

  LowRank& LowRank::operator=(LowRank&& A) = default;

  std::unique_ptr<Node> LowRank::clone() const {
    return std::make_unique<LowRank>(*this);
  }

  std::unique_ptr<Node> LowRank::move_clone() {
    return std::make_unique<LowRank>(std::move(*this));
  }

  const char* LowRank::type() const { return "LowRank"; }

  LowRank::LowRank(const Node& node, int k)
  : Node(node), dim{row_range.length, col_range.length}, rank(k) {
    MM_INIT();
    U = Dense(dim[0], k, i_abs, j_abs, level);
    S = Dense(k, k, i_abs, j_abs, level);
    V = Dense(k, dim[1], i_abs, j_abs, level);
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
    MM_INIT();
    // Rank with oversampling limited by dimensions
    rank = std::min(std::min(k+5, dim[0]), dim[1]);
    std::tie(U, S, V) = rsvd(A, rank);
    U.resize(dim[0], k);
    S.resize(k, k);
    V.resize(k, dim[1]);
    rank = k;
  }
  std::tuple<Dense, Dense> merge_basis(
    const Dense& A, const Dense& B, bool trans
  ) {
    // For U trans=false
    // For V trans=true
    assert(A.dim[trans?1:0] == B.dim[trans?1:0]);
    int N = A.dim[trans?1:0];
    int Arank = A.dim[trans?0:1];
    int Brank = B.dim[trans?0:1];
    Dense AB(Arank, Brank);
    gemm(A, B, AB, !trans, trans, 1, 0);

    Dense AAB(N, Brank);
    gemm(A, AB, AAB, trans, false, 1, 0);

    Dense B_AAB(N, Brank);
    B_AAB = (trans ? B.transpose(): B) - AAB;

    Dense Q(N, Brank);
    Dense R(Brank, Brank);
    qr(B_AAB, Q, R);

    // Some copies can be avoided here once H(D) w/o copies exists
    Hierarchical InnerH(2, 1);
    InnerH[0] = std::move(AB);
    InnerH[1] = std::move(R);
    Dense Inner(InnerH);

    if (trans) Q.transpose();
    return {std::move(Q), std::move(Inner)};
  }

  std::tuple<Dense, Dense, Dense> merge_S(
    const Dense& S, const Dense& AS,
    const Dense& InnerU, const Dense& InnerVt
  ) {
    Dense InnerUAS(InnerU.dim[0], AS.dim[1]);
    gemm(InnerU, AS, InnerUAS, 1, 0);

    // TODO consider copies here, especially once H(D) no longer copies!
    // Also consider move!
    Hierarchical M(2, 2);
    M(0, 0) = S;
    M(0, 1) = Dense(S.dim[0], S.dim[1]);
    M(1, 0) = Dense(S.dim[0], S.dim[1]);
    M(1, 1) = Dense(S.dim[0], S.dim[1]);
    Dense MD(M);
    gemm(InnerUAS, InnerVt, MD, false, true, 1, 1);

    Dense Uhat, Shat, Vhat;
    std::tie(Uhat, Shat, Vhat) = svd(MD);
    Shat.resize(S.dim[0], S.dim[1]);
    Uhat.resize(Uhat.dim[0], S.dim[0]);
    Vhat.resize(S.dim[1], Vhat.dim[1]);

    return {std::move(Uhat), std::move(Shat), std::move(Vhat)};
  }

  const LowRank& LowRank::operator+=(const LowRank& A) {
    assert(dim[0] == A.dim[0]);
    assert(dim[1] == A.dim[1]);
    assert(rank == A.rank);
    if(getCounter("LR_ADDITION_COUNTER") == 1) updateCounter("LR-addition", 1);
    if(getCounter("LRA") == 0) {
      //Truncate and Recompress if rank > min(nrow, ncol)
      if (rank+A.rank >= std::min(dim[0], dim[1])) {
        *this = LowRank(Dense(*this) + Dense(A), rank);
      }
      else {
        LowRank B(dim[0], dim[1], rank+A.rank, i_abs, j_abs, level);
        B.mergeU(*this, A);
        B.mergeS(*this, A);
        B.mergeV(*this, A);
        rank += A.rank;
        U = std::move(B.U);
        S = std::move(B.S);
        V = std::move(B.V);
      }
    } else if(getCounter("LRA") == 1) {
      //Bebendorf HMatrix Book p16
      //Rounded Addition
      LowRank B(dim[0], dim[1], rank+A.rank, i_abs, j_abs, level);
      B.mergeU(*this, A);
      B.mergeS(*this, A);
      B.mergeV(*this, A);

      Dense BU_copy(B.U);
      gemm(BU_copy, B.S, B.U, 1, 0);

      Dense Qu(B.U.dim[0], B.U.dim[1]);
      Dense Ru(B.U.dim[1], B.U.dim[1]);
      qr(B.U, Qu, Ru);

      B.V.transpose();
      Dense Qv(B.V.dim[0], B.V.dim[1]);
      Dense Rv(B.V.dim[1], B.V.dim[1]);
      qr(B.V, Qv, Rv);

      Dense RuRvT(Ru.dim[0], Rv.dim[0]);
      gemm(Ru, Rv, RuRvT, false, true, 1, 0);

      Dense RRU, RRS, RRV;
      std::tie(RRU, RRS, RRV) = svd(RuRvT);

      RRS.resize(rank, rank);
      S = std::move(RRS);
      RRU.resize(RRU.dim[0], rank);
      gemm(Qu, RRU, U, 1, 0);
      RRV.resize(rank, RRV.dim[1]);
      gemm(RRV, Qv, V, false, true, 1, 0);
    } else {
      //Bebendorf HMatrix Book p17
      //Rounded addition by exploiting orthogonality

      // TODO consider copies here, especially once H(D) no longer copies!
      Hierarchical OuterU(1, 2);
      Dense InnerU;
      std::tie(OuterU[1], InnerU) = merge_basis(U, A.U, false);
      OuterU[0] = std::move(U);

      Hierarchical OuterV(2, 1);
      Dense InnerVt;
      std::tie(OuterV[1], InnerVt) = merge_basis(V, A.V, true);
      OuterV[0] = std::move(V);

      Dense Uhat, Vhat;
      std::tie(Uhat, S, Vhat) = merge_S(S, A.S, InnerU, InnerVt);

      // Restore moved-from U and V and finalize basis
      U = Dense(dim[0], rank);
      V = Dense(rank, dim[1]);
      gemm(OuterU, Uhat, U, 1, 0);
      gemm(Vhat, OuterV, V, 1, 0);
    }
    return *this;
  }

  void LowRank::mergeU(const LowRank& A, const LowRank& B) {
    assert(rank == A.rank + B.rank);
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<A.rank; j++) {
        U(i,j) = A.U(i,j);
      }
      for (int j=0; j<B.rank; j++) {
        U(i,j+A.rank) = B.U(i,j);
      }
    }
  }

  void LowRank::mergeS(const LowRank& A, const LowRank& B) {
    assert(rank == A.rank + B.rank);
    for (int i=0; i<A.rank; i++) {
      for (int j=0; j<A.rank; j++) {
        S(i,j) = A.S(i,j);
      }
      for (int j=0; j<B.rank; j++) {
        S(i,j+A.rank) = 0;
      }
    }
    for (int i=0; i<B.rank; i++) {
      for (int j=0; j<A.rank; j++) {
        S(i+A.rank,j) = 0;
      }
      for (int j=0; j<B.rank; j++) {
        S(i+A.rank,j+A.rank) = B.S(i,j);
      }
    }
  }

  void LowRank::mergeV(const LowRank& A, const LowRank& B) {
    assert(rank == A.rank + B.rank);
    for (int i=0; i<A.rank; i++) {
      for (int j=0; j<dim[1]; j++) {
        V(i,j) = A.V(i,j);
      }
    }
    for (int i=0; i<B.rank; i++) {
      for (int j=0; j<dim[1]; j++) {
        V(i+A.rank,j) = B.V(i,j);
      }
    }
  }

  LowRank LowRank::get_part(const Node& node) const {
    assert(is_child(node));
    LowRank A(node, rank);
    int rel_row_begin = A.row_range.start - row_range.start;
    int rel_col_begin = A.col_range.start - col_range.start;
    for (int i=0; i<A.U.dim[0]; i++) {
      for (int k=0; k<A.U.dim[1]; k++) {
        A.U(i, k) = U(i+rel_row_begin, k);
      }
    }
    A.S = S;
    for (int k=0; k<A.V.dim[0]; k++) {
      for (int j=0; j<A.V.dim[1]; j++) {
        A.V(k, j) = V(k, j+rel_col_begin);
      }
    }
    return A;
  }

} // namespace hicma
