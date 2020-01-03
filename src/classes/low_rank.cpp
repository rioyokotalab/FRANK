#include "hicma/classes/low_rank.h"

#include "hicma/classes/node.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/LAPACK/qr.h"
#include "hicma/operations/randomized/rsvd.h"
#include "hicma/util/print.h"
#include "hicma/util/counter.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <random>
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

  LowRank::LowRank(
    int m, int n,
    int k,
    int i_abs, int j_abs,
    int level
  ) : Node(i_abs, j_abs, level), dim{m, n}, rank(k) {
    MM_INIT();
    U = Dense(m, k, i_abs, j_abs, level);
    S = Dense(k, k, i_abs, j_abs, level);
    V = Dense(k, n, i_abs, j_abs, level);
  }

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

      Dense RRU(RuRvT.dim[0], RuRvT.dim[0]);
      Dense RRS(RuRvT.dim[0], RuRvT.dim[1]);
      Dense RRV(RuRvT.dim[1], RuRvT.dim[1]);
      RuRvT.svd(RRU, RRS, RRV);

      RRS.resize(rank, rank);
      S = std::move(RRS);
      RRU.resize(RRU.dim[0], rank);
      gemm(Qu, RRU, U, 1, 0);
      RRV.resize(rank, RRV.dim[1]);
      gemm(RRV, Qv, V, false, true, 1, 0);
    } else {
      //Bebendorf HMatrix Book p17
      //Rounded addition by exploiting orthogonality
      int rank2 = 2 * rank;

      Dense Xu(rank, rank);
      gemm(U, A.U, Xu, true, false, 1, 0);

      Dense Ua(A.dim[0], rank);
      Dense Yu(A.dim[0], rank);
      gemm(U, Xu, Ua, 1, 0);

      Yu = A.U - Ua;

      Dense Qu(dim[0], rank);
      Dense Ru(rank, rank);
      qr(Yu, Qu, Ru);

      Dense Xv(rank, rank);
      gemm(V, A.V, Xv, false, true, 1, 0);

      Dense Va_Xv(dim[1], rank);
      gemm(V, Xv, Va_Xv, true, false, 1, 0);

      Dense Yv(dim[1], rank);
      Dense VB = A.V.transpose();
      Yv = VB - Va_Xv;

      Dense Qv(dim[1], rank);
      Dense Rv(rank, rank);
      qr(Yv, Qv, Rv);

      Hierarchical M(2, 2);
      Dense Xu_BS(rank, rank);
      gemm(Xu, A.S, Xu_BS, 1, 0);
      Dense Ru_BS(rank, rank);
      gemm(Ru, A.S, Ru_BS, 1, 0);

      gemm(Xu_BS, Xv, S, false, true, 1, 1);
      M(0,0) = S;
      gemm(Xu_BS, Rv, S, false, true, 1, 0);
      M(0,1) = S;
      gemm(Ru_BS, Xv, S, false, true, 1, 0);
      M(1,0) = S;
      gemm(Ru_BS, Rv, S, false, true, 1, 0);
      M(1,1) = S;

      Dense Uhat(rank2, rank2);
      Dense Shat(rank2, rank2);
      Dense Vhat(rank2, rank2);
      Dense(M).svd(Uhat, Shat, Vhat);

      Uhat.resize(rank2, rank);
      Shat.resize(rank, rank);
      Vhat.resize(rank, rank2);

      Dense MERGE_U(dim[0], rank2);
      Dense MERGE_V(dim[1], rank2);

      for (int i = 0; i < dim[0]; ++i) {
        for (int j = 0; j < rank; ++j) {
          MERGE_U(i,j) = U(i,j);
        }
        for (int j = 0; j < rank; ++j) {
          MERGE_U(i, rank + j) = Qu(i,j);
        }
      }

      for (int i = 0; i < dim[1]; ++i) {
        for (int j = 0; j < rank; ++j) {
          MERGE_V(i, j) = V(j,i);
        }
        for (int j = 0; j < rank; ++j) {
          MERGE_V(i, j + rank) = Qv(i,j);
        }
      }

      gemm(MERGE_U, Uhat, U, 1, 0);
      S = std::move(Shat);
      gemm(Vhat, MERGE_V, V, false, true, 1, 0);
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

} // namespace hicma
