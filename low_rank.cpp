#include "low_rank.h"

namespace hicma {
  LowRank::LowRank() {
    dim[0]=0; dim[1]=0; rank=0;
  }

  LowRank::LowRank(const int m, const int n, const int k) {
    dim[0]=m; dim[1]=n; rank=k;
    U = DensePtr(m,k);
    S = DensePtr(k,k);
    V = DensePtr(k,n);
  }

  LowRank::LowRank(const LowRank &A) : _Node(A.i_abs,A.j_abs,A.level), U(A.U), S(A.S), V(A.V) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
  }

  LowRank::LowRank(const LowRank *A) : _Node(A->i_abs,A->j_abs,A->level), U(A->U), S(A->S), V(A->V) {
    dim[0]=A->dim[0]; dim[1]=A->dim[1]; rank=A->rank;
  }

  LowRank::LowRank(const Dense &A, const int k) : _Node(A.i_abs,A.j_abs,A.level) {
    int m = dim[0] = A.dim[0];
    int n = dim[1] = A.dim[1];
    rank = k;
    U = DensePtr(m,k);
    S = DensePtr(k,k);
    V = DensePtr(k,n);
    randomized_low_rank_svd2(A.data, rank, (*U).data, (*S).data, (*V).data, m, n);
  }

  LowRank::LowRank(
               const Node A,
               const int k
               ) : _Node(A->i_abs,A->j_abs,A->level){
    assert(A.is(HICMA_DENSE));
    const Dense& AR = static_cast<Dense&>(*A);
    int m = dim[0] = AR.dim[0];
    int n = dim[1] = AR.dim[1];
    rank = k;
    U = DensePtr(m,k);
    S = DensePtr(k,k);
    V = DensePtr(k,n);
    randomized_low_rank_svd2(AR.data, rank, (*U).data, (*S).data, (*V).data, m, n);
  }

  LowRank* LowRank::clone() const {
    LowRank* Out =  new LowRank(*this);
    // The following lines are necessary since the constructor from a LowRank&
    // does not do a deep copy
    Out->U = U->clone();
    Out->S = S->clone();
    Out->V = V->clone();
    return Out;
  }

  const bool LowRank::is(const int enum_id) const {
    return enum_id == HICMA_LOWRANK;
  }

  const char* LowRank::is_string() const { return "LowRank"; }

  const _Node& LowRank::operator=(const double a) {
    assert(a == 0);
    U = 0;
    S = 0;
    V = 0;
    return *this;
  }

  const _Node& LowRank::operator=(const _Node& A) {
    if (A.is(HICMA_LOWRANK)) {
      const LowRank& AR = static_cast<const LowRank&>(A);
      assert(dim[0]==AR.dim[0] && dim[1]==AR.dim[1] && rank==AR.rank);
      dim[0]=AR.dim[0]; dim[1]=AR.dim[1]; rank=AR.rank;
      U = AR.U;
      S = AR.S;
      V = AR.V;
      return *this;
    } else {
      std::cout << this->is_string() << " = " << A.is_string();
      std::cout << " not implemented!" << std::endl;
      return *this;
    }
  }

  const _Node& LowRank::operator=(const Node& A) {
    return *this = *A;
  }

  LowRank LowRank::operator-() const {
    LowRank A(*this);
    A.U = -U;
    A.S = -S;
    A.V = -V;
    return A;
  }

  Node LowRank::add(const Node& B) const {
    if (B.is(HICMA_LOWRANK)) {
      const LowRank& BR = static_cast<const LowRank&>(*B);
      assert(dim[0]==BR.dim[0] && dim[1]==BR.dim[1]);
      LowRankPtr Out;
      if (rank+BR.rank >= dim[0]) {
        Out = LowRankPtr((*this).dense() + BR.dense(), rank);
      }
      else {
        Out = LowRankPtr(dim[0], dim[1], rank+BR.rank);
        (*Out).mergeU(*this,BR);
        (*Out).mergeS(*this,BR);
        (*Out).mergeV(*this,BR);
      }
      return Out;
    } else if(B.is(HICMA_DENSE)) {
      const Dense& BR = static_cast<const Dense&>(*B);
      assert(dim[0]==BR.dim[0] && dim[1]==BR.dim[1]);
      return this->dense()->add(B);
    } else {
      std::cout << this->is_string() << " + " << B.is_string();
      std::cout << " is undefined!" << std::endl;
      return Node(nullptr);
    }
  }

  Node LowRank::sub(const Node& B) const {
    if (B.is(HICMA_LOWRANK)) {
      const LowRank& BR = static_cast<const LowRank&>(*B);
      assert(dim[0]==BR.dim[0] && dim[1]==BR.dim[1]);
      LowRankPtr Out;
      if (rank+BR.rank >= dim[0]) {
        Out = LowRankPtr((*this).dense() - BR.dense(), rank);
      }
      else {
        Out = LowRankPtr(dim[0], dim[1], rank+BR.rank);
        (*Out).mergeU(*this,-BR);
        (*Out).mergeS(*this,-BR);
        (*Out).mergeV(*this,-BR);
      }
      return Out;
    } else if(B.is(HICMA_DENSE)) {
      const Dense& BR = static_cast<const Dense&>(*B);
      assert(dim[0]==BR.dim[0] && dim[1]==BR.dim[1]);
      return this->dense()->sub(B);
    } else {
      std::cout << this->is_string() << " - " << B.is_string();
      std::cout << " is undefined!" << std::endl;
      return Node(nullptr);
    }
  }

  Node LowRank::mul(const Node& B) const {
    if (B.is(HICMA_LOWRANK)) {
      const LowRank& BR = static_cast<const LowRank&>(*B);
      assert(dim[1] == BR.dim[0]);
      LowRankPtr Out(dim[0], BR.dim[1], rank);
      (*Out).U = U;
      (*Out).S = S * (V * BR.U) * BR.S;
      (*Out).V = BR.V;
      return Out;
    } else if(B.is(HICMA_DENSE)) {
      const Dense& BR = static_cast<const Dense&>(*B);
      assert(dim[1] == BR.dim[0]);
      LowRankPtr Out(dim[0], BR.dim[1], rank);
      (*Out).U = U;
      (*Out).S = S;
      (*Out).V = V->mul(B);
      return Out;
    } else {
      std::cout << this->is_string() << " * " << B.is_string();
      std::cout << " is undefined!" << std::endl;
      return Node(nullptr);
    }
  }

  void LowRank::resize(int m, int n, int k) {
    dim[0]=m; dim[1]=n; rank=k;
    U.resize(m,k);
    S.resize(k,k);
    V.resize(k,n);
  }

  const Node LowRank::dense() const {
    return U * S * V;
  }

  double LowRank::norm() const {
    return this->dense()->norm();
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

  void LowRank::trsm(const Node& A, const char& uplo) {
    if (A.is(HICMA_DENSE)) {
      switch (uplo) {
      case 'l' :
        U.trsm(A, uplo);
        break;
      case 'u' :
        V.trsm(A, uplo);
        break;
      }
    } else {
      fprintf(
          stderr,"%s /= %s undefined.\n",
          this->is_string(), A.is_string());
      abort();
    }
  }

  void LowRank::gemm(const Node& A, const Node& B) {
    if (A.is(HICMA_DENSE)) {
      if (B.is(HICMA_DENSE)) {
        fprintf(
            stderr,"%s += %s * %s undefined.\n",
            this->is_string(), A.is_string(), B.is_string());
        abort();
      } else if (B.is(HICMA_LOWRANK)) {
        *this = this->sub(A * B);
      } else if (B.is(HICMA_HIERARCHICAL)) {
        fprintf(
            stderr,"%s += %s * %s undefined.\n",
            this->is_string(), A.is_string(), B.is_string());
        abort();
      }
    } else if (A.is(HICMA_LOWRANK)) {
      if (B.is(HICMA_DENSE)) {
        *this = this->sub(A * B);
      } else if (B.is(HICMA_LOWRANK)) {
        *this = this->sub(A * B);
      } else if (B.is(HICMA_HIERARCHICAL)) {
        fprintf(
            stderr,"%s += %s * %s undefined.\n",
            this->is_string(), A.is_string(), B.is_string());
        abort();
      }
    } else {
      fprintf(
          stderr,"%s += %s * %s undefined.\n",
          this->is_string(), A.is_string(), B.is_string());
      abort();
    }
  }
}
