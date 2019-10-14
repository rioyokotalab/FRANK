#include "hicma/node_proxy.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/operations.h"
#include "hicma/util/print.h"

#include <cassert>
#include <iostream>
#include <random>
#include <algorithm>

#include "yorel/multi_methods.hpp"

namespace hicma {

  LowRank::LowRank() {
    MM_INIT();
    dim[0]=0; dim[1]=0; rank=0;
  }

  LowRank::LowRank(
                   const int m,
                   const int n,
                   const int k,
                   const int i_abs,
                   const int j_abs,
                   const int level) {
    MM_INIT();
    dim[0]=m; dim[1]=n; rank=k;
    U = Dense(m, k, i_abs, j_abs, level);
    S = Dense(k, k, i_abs, j_abs, level);
    V = Dense(k, n, i_abs, j_abs, level);
  }

  LowRank::LowRank(const Dense& A, const int k) : Node(A.i_abs,A.j_abs,A.level) {
    MM_INIT();
    dim[0] = A.dim[0];
    dim[1] = A.dim[1];
    // Rank with oversampling limited by dimensions
    rank = std::min(std::min(k+5, dim[0]), dim[1]);
    U = Dense(dim[0], k, i_abs, j_abs, level);
    S = Dense(rank, rank, i_abs, j_abs, level);
    V = Dense(k, dim[1], i_abs, j_abs, level);
    Dense RN(dim[1],rank);
    std::mt19937 generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i=0; i<dim[1]*rank; i++) {
      RN[i] = distribution(generator); // RN = randn(n,k+p)
    }
    Dense Y(dim[0],rank);
    gemm(A, RN, Y, CblasNoTrans, CblasNoTrans, 1, 0); // Y = A * RN
    Dense Q(dim[0],rank);
    Dense R(rank,rank);
    Y.qr(Q, R); // [Q, R] = qr(Y)
    Dense Bt(dim[1],rank);
    gemm(A, Q, Bt, CblasTrans, CblasNoTrans, 1, 0); // B' = A' * Q
    Dense Qb(dim[1],rank);
    Dense Rb(rank,rank);
    Bt.qr(Qb,Rb); // [Qb, Rb] = qr(B')
    Dense Ur(rank,rank);
    Dense Vr(rank,rank);
    Rb.svd(Vr,S,Ur); // [Vr, S, Ur] = svd(Rb);
    Ur.resize(k,rank);
    gemm(Q, Ur, U, CblasNoTrans, CblasTrans, 1, 0); // U = Q * Ur'
    Vr.resize(rank,k);
    gemm(Vr, Qb, V, CblasTrans, CblasTrans, 1, 0); // V = Vr' * Qb'
    S.resize(k,k);
    rank = k;
  }

  LowRank::LowRank(const LowRank& A) : Node(A.i_abs,A.j_abs,A.level), U(A.U), S(A.S), V(A.V) {
    MM_INIT();
    dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
  }

  LowRank::LowRank(LowRank&& A) {
    MM_INIT();
    swap(*this, A);
  }

  // LowRank::LowRank(const NodeProxy& _A, const int k) : Node(_A.ptr->i_abs, _A.ptr->j_abs, _A.ptr->level) {
  //   if (_A.is(HICMA_LOWRANK)) {
  //     *this = static_cast<const LowRank&>(*_A.ptr);
  //   } else if (_A.is(HICMA_DENSE)) {
  //     *this = LowRank(static_cast<const Dense&>(*_A.ptr), k);
  //   } else if (_A.is(HICMA_HIERARCHICAL)) {
  //     std::cerr << this->type() << "(" << _A.type();
  //     std::cerr << ") undefined." << std::endl;
  //     abort();
  //   }
  // }

  LowRank* LowRank::clone() const {
    return new LowRank(*this);
  }

  void swap(LowRank& A, LowRank& B) {
    using std::swap;
    swap(A.U, B.U);
    swap(A.S, B.S);
    swap(A.V, B.V);
    swap(A.dim, B.dim);
    swap(A.i_abs, B.i_abs);
    swap(A.j_abs, B.j_abs);
    swap(A.level, B.level);
    swap(A.rank, B.rank);
  }

  const LowRank& LowRank::operator=(LowRank A) {
    swap(*this, A);
    return *this;
  }

  bool LowRank::is(const int enum_id) const {
    return enum_id == HICMA_LOWRANK;
  }

  const LowRank& LowRank::operator+=(const LowRank& A) {
    assert(dim[0] == A.dim[0]);
    assert(dim[1] == A.dim[1]);
    assert(rank == A.rank);
#if 0
    LowRank B(dim[0], dim[1], rank+A.rank, i_abs, j_abs, level);
    B.mergeU(*this, A);
    B.mergeS(*this, A);
    B.mergeV(*this, A);
    rank += A.rank;
    swap(U, B.U);
    swap(S, B.S);
    swap(V, B.V);
#else
    int rank2 = 2 * rank;

    Dense Xu(rank, rank);
    gemm(U, A.U, Xu, CblasTrans, CblasNoTrans, 1, 0);

    Dense Ua(A.dim[0], rank);
    Dense Yu(A.dim[0], rank);
    gemm(U, Xu, Ua, 1, 0);

    Yu = A.U - Ua;

    Dense Qu(dim[0], rank);
    Dense Ru(rank, rank);
    Yu.qr(Qu, Ru);

    Dense Xv(rank, rank);
    gemm(V, A.V, Xv, CblasNoTrans, CblasTrans, 1, 0);

    Dense Va_Xv(dim[1], rank);
    gemm(V, Xv, Va_Xv, CblasTrans, CblasNoTrans, 1, 0);

    Dense Yv(dim[1], rank);
    Dense VB = A.V.transpose();
    Yv = VB - Va_Xv;

    Dense Qv(dim[1], rank);
    Dense Rv(rank, rank);
    Yv.qr(Qv, Rv);

    Hierarchical M(2, 2);
    Dense Xu_BS(rank, rank);
    gemm(Xu, A.S, Xu_BS, 1, 0);
    Dense Ru_BS(rank, rank);
    gemm(Ru, A.S, Ru_BS, 1, 0);

    gemm(Xu_BS, Xv, S, CblasNoTrans, CblasTrans, 1, 1);
    M(0,0) = S;
    gemm(Xu_BS, Rv, S, CblasNoTrans, CblasTrans, 1, 0);
    M(0,1) = S;
    gemm(Ru_BS, Xv, S, CblasNoTrans, CblasTrans, 1, 0);
    M(1,0) = S;
    gemm(Ru_BS, Rv, S, CblasNoTrans, CblasTrans, 1, 0);
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
    swap(S, Shat);
    gemm(Vhat, MERGE_V, V, CblasNoTrans, CblasTrans, 1, 0);
#endif
    return *this;
  }

  const char* LowRank::type() const { return "LowRank"; }

  double LowRank::norm() const {
    return Dense(*this).norm();
  }

  void LowRank::print() const {
    std::cout << "U : ------------------------------------------------------------------------------" << std::endl;
    U.print();
    std::cout << "S : ------------------------------------------------------------------------------" << std::endl;
    S.print();
    std::cout << "V : ------------------------------------------------------------------------------" << std::endl;
    V.print();
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
  }

  void LowRank::transpose() {
    using std::swap;
    U.transpose();
    S.transpose();
    V.transpose();
    swap(dim[0], dim[1]);
    swap(U, V);
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

}
