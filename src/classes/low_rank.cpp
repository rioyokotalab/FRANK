#include "hicma/classes/low_rank.h"

#include "hicma/classes/node.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/LAPACK/qr.h"
#include "hicma/operations/LAPACK/svd.h"
#include "hicma/operations/randomized/rsvd.h"
#include "hicma/util/counter.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <tuple>
#include <utility>

#include "yorel/multi_methods.hpp"

namespace hicma {

  LowRank::LowRank() : dim{0, 0}, rank(0) { MM_INIT(); }

  LowRank::~LowRank() = default;

  LowRank::LowRank(const LowRank& A)
  : Node(A), _U(A.U()), _S(A.S()), _V(A.V()),
    dim{A.dim[0], A.dim[1]}, rank(A.rank)
  {
    MM_INIT();
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

  Dense& LowRank::U() { return _U; }
  const Dense& LowRank::U() const { return _U; }

  Dense& LowRank::S() { return _S; }
  const Dense& LowRank::S() const { return _S; }

  Dense& LowRank::V() { return _V; }
  const Dense& LowRank::V() const { return _V; }

  LowRank::LowRank(const Node& node, int k, bool node_only)
  : Node(node), dim{row_range.length, col_range.length}, rank(k) {
    MM_INIT();
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
    MM_INIT();
    // Rank with oversampling limited by dimensions
    rank = std::min(std::min(k+5, dim[0]), dim[1]);
    std::tie(U(), S(), V()) = rsvd(A, rank);
    U().resize(dim[0], k);
    S().resize(k, k);
    V().resize(k, dim[1]);
    rank = k;
  }

  std::tuple<Dense, Dense> merge_col_basis(
    const Dense& U, const Dense& Au
  ) {
    assert(U.dim[0] == Au.dim[0]);
    int Arank = U.dim[1];
    int Brank = Au.dim[1];
    assert(Arank == Brank);

    Dense InnerU(Arank+Brank, Brank);
    NoCopySplit InnerH(InnerU, 2, 1);
    gemm(U, Au, InnerH[0], true, false, 1, 0);

    Dense U_UUtAu(Au);
    gemm(U, InnerH[0], U_UUtAu, -1, 1);

    Dense Q(U.dim[0], Brank);
    qr(U_UUtAu, Q, InnerH[1]);

    return {std::move(Q), std::move(InnerU)};
  }

  std::tuple<Dense, Dense> merge_row_basis(
    const Dense& V, const Dense& Av
  ) {
    assert(V.dim[1] == Av.dim[1]);
    int Arank = V.dim[0];
    int Brank = Av.dim[0];
    assert(Arank == Brank);

    Dense InnerV(Brank, Arank+Brank);
    NoCopySplit InnerH(InnerV, 1, 2);
    gemm(Av, V, InnerH[0], false, true, 1, 0);

    Dense Av_AvVtV(Av);
    gemm(InnerH[0], V, Av_AvVtV, -1, 1);

    Dense Q(Brank, V.dim[1]);
    rq(Av_AvVtV, InnerH[1], Q);

    return {std::move(Q), std::move(InnerV)};
  }


  std::tuple<Dense, Dense, Dense> merge_S(
    const Dense& S, const Dense& AS,
    const Dense& InnerU, const Dense& InnerV
  ) {
    assert(S.dim[0] == S.dim[1]);
    int rank = S.dim[0];

    Dense InnerUAS(InnerU.dim[0], AS.dim[1]);
    gemm(InnerU, AS, InnerUAS, 1, 0);

    // TODO Consider using move for S if possible!
    Dense M(rank*2, rank*2);
    for (int i=0; i<rank; i++) {
      for (int j=0; j<rank; j++) {
        M(i, j) = S(i, j);
      }
    }
    gemm(InnerUAS, InnerV, M, 1, 1);

    Dense Uhat, Shat, Vhat;
    std::tie(Uhat, Shat, Vhat) = svd(M);

    Shat.resize(rank, rank);
    Uhat.resize(Uhat.dim[0], rank);
    Vhat.resize(rank, Vhat.dim[1]);

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
        U() = std::move(B.U());
        S() = std::move(B.S());
        V() = std::move(B.V());
      }
    } else if(getCounter("LRA") == 1) {
      //Bebendorf HMatrix Book p16
      //Rounded Addition
      LowRank B(dim[0], dim[1], rank+A.rank, i_abs, j_abs, level);
      B.mergeU(*this, A);
      B.mergeS(*this, A);
      B.mergeV(*this, A);

      Dense BU_copy(B.U());
      gemm(BU_copy, B.S(), B.U(), 1, 0);

      Dense Qu(B.U().dim[0], B.U().dim[1]);
      Dense Ru(B.U().dim[1], B.U().dim[1]);
      qr(B.U(), Qu, Ru);

      B.V().transpose();
      Dense Qv(B.V().dim[0], B.V().dim[1]);
      Dense Rv(B.V().dim[1], B.V().dim[1]);
      qr(B.V(), Qv, Rv);

      Dense RuRvT(Ru.dim[0], Rv.dim[0]);
      gemm(Ru, Rv, RuRvT, false, true, 1, 0);

      Dense RRU, RRS, RRV;
      std::tie(RRU, RRS, RRV) = svd(RuRvT);

      RRS.resize(rank, rank);
      S() = std::move(RRS);
      RRU.resize(RRU.dim[0], rank);
      gemm(Qu, RRU, U(), 1, 0);
      RRV.resize(rank, RRV.dim[1]);
      gemm(RRV, Qv, V(), false, true, 1, 0);
    } else {
      //Bebendorf HMatrix Book p17
      //Rounded addition by exploiting orthogonality
      timing::start("LR += LR");

      // TODO consider copies here, especially once H(D) no longer copies!
      timing::start("Merge col basis");
      Hierarchical OuterU(1, 2);
      Dense InnerU;
      std::tie(OuterU[1], InnerU) = merge_col_basis(U(), A.U());
      OuterU[0] = std::move(U());
      timing::stop("Merge col basis");

      timing::start("Merge row basis");
      Hierarchical OuterV(2, 1);
      Dense InnerVt;
      std::tie(OuterV[1], InnerVt) = merge_row_basis(V(), A.V());
      OuterV[0] = std::move(V());
      timing::stop("Merge row basis");

      timing::start("Merge S");
      Dense Uhat, Vhat;
      std::tie(Uhat, S(), Vhat) = merge_S(S(), A.S(), InnerU, InnerVt);
      timing::stop("Merge S");

      // Restore moved-from U and V and finalize basis
      U() = Dense(dim[0], rank);
      V() = Dense(rank, dim[1]);
      gemm(OuterU, Uhat, U(), 1, 0);
      gemm(Vhat, OuterV, V(), 1, 0);

      timing::stop("LR += LR");
    }
    return *this;
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


  LowRankView::LowRankView() : LowRank() { MM_INIT(); }

  LowRankView::~LowRankView() = default;

  LowRankView::LowRankView(const LowRankView& A) {
    MM_INIT();
    *this = A;
  }
  LowRankView& LowRankView::operator=(const LowRankView& A) = default;

  LowRankView::LowRankView(LowRankView&& A) {
    MM_INIT();
    *this = std::move(A);
  }

  LowRankView& LowRankView::operator=(LowRankView&& A) = default;

  std::unique_ptr<Node> LowRankView::clone() const {
    return std::make_unique<LowRankView>(*this);
  }

  std::unique_ptr<Node> LowRankView::move_clone() {
    return std::make_unique<LowRankView>(std::move(*this));
  }

  const char* LowRankView::type() const {
    return "LowRankView";
  }

  // TODO write safe setters!
  // A.U() = B is a shitty way to write things. A.setU(B) is better.
  DenseView& LowRankView::U() { return _U; }
  const DenseView& LowRankView::U() const { return _U; }

  DenseView& LowRankView::S() { return _S; }
  const DenseView& LowRankView::S() const { return _S; }

  DenseView& LowRankView::V() { return _V; }
  const DenseView& LowRankView::V() const { return _V; }

  LowRankView::LowRankView(const Node& node, const LowRank& A)
  : LowRank(node, A.rank, true) {
    MM_INIT();
    int rel_row_start = (
      node.row_range.start-A.row_range.start + A.U().row_range.start);
    U() = DenseView(Node(
      0, 0, A.U().level+1,
      IndexRange(rel_row_start, node.row_range.length),
      IndexRange(0, A.rank)
    ), A.U());
    S() = DenseView(
      Node(0, 0, A.S().level+1, IndexRange(0, A.rank), IndexRange(0, A.rank)),
      A.S()
    );
    int rel_col_start = (
      node.col_range.start-A.col_range.start + A.V().col_range.start);
    V() = DenseView(Node(
      0, 0, A.V().level+1,
      IndexRange(0, A.rank),
      IndexRange(rel_col_start, node.col_range.length)
    ), A.V());
  }
} // namespace hicma
