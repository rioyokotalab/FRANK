#include "low_rank.h"

#include "dense.h"
#include "id.h"

#include <cstdio>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>

namespace hicma {
  LowRank::LowRank() {
    dim[0]=0; dim[1]=0; rank=0;
  }

  LowRank::LowRank(const int m, const int n, const int k) {
    dim[0]=m; dim[1]=n; rank=k;
    U = Dense(m,k);
    S = Dense(k,k);
    V = Dense(k,n);
  }

  LowRank::LowRank(const Dense& A, const int k) : Node(A.i_abs,A.j_abs,A.level) {
    int m = dim[0] = A.dim[0];
    int n = dim[1] = A.dim[1];
    rank = k;
    U = Dense(m,k);
    S = Dense(k,k);
    V = Dense(k,n);
    randomized_low_rank_svd2(A.data, rank, U.data, S.data, V.data, m, n);
  }

  // TODO: This constructor is called A LOT. Work that out
  LowRank::LowRank(
               const Block& A,
               const int k
               ) : Node(A.ptr->i_abs,A.ptr->j_abs,A.ptr->level){
    assert(A.is(HICMA_DENSE));
    const Dense& AR = static_cast<Dense&>(*A.ptr);
    int m = dim[0] = AR.dim[0];
    int n = dim[1] = AR.dim[1];
    rank = k;
    U = Dense(m,k);
    S = Dense(k,k);
    V = Dense(k,n);
    randomized_low_rank_svd2(AR.data, rank, U.data, S.data, V.data, m, n);
  }

  LowRank::LowRank(const LowRank& A) : Node(A.i_abs,A.j_abs,A.level), U(A.U), S(A.S), V(A.V) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
  }

  LowRank::LowRank(LowRank&& A) {
    swap(*this, A);
  }

  LowRank::LowRank(const LowRank* A) : Node(A->i_abs,A->j_abs,A->level), U(A->U), S(A->S), V(A->V) {
    dim[0]=A->dim[0]; dim[1]=A->dim[1]; rank=A->rank;
  }

  LowRank* LowRank::clone() const {
    return new LowRank(*this);
  }

  void swap(LowRank& first, LowRank& second) {
    using std::swap;
    swap(first.U, second.U);
    swap(first.S, second.S);
    swap(first.V, second.V);
    swap(first.dim, second.dim);
    swap(first.i_abs, second.i_abs);
    swap(first.j_abs, second.j_abs);
    swap(first.level, second.level);
    swap(first.rank, second.rank);
  }

  const Node& LowRank::operator=(const Node& A) {
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

  const Node& LowRank::operator=(Node&& A) {
    if (A.is(HICMA_LOWRANK)) {
      swap(*this, static_cast<LowRank&>(A));
      return *this;
    } else {
      std::cout << this->is_string() << " = " << A.is_string();
      std::cout << " not implemented!" << std::endl;
      return *this;
    }
  }

  const LowRank& LowRank::operator=(LowRank A) {
    swap(*this, A);
    return *this;
  }

  const Node& LowRank::operator=(Block A) {
    return *this = std::move(*A.ptr);
  }

  const Node& LowRank::operator=(const double a) {
    assert(a == 0);
    U = 0;
    S = 0;
    V = 0;
    return *this;
  }

  LowRank LowRank::operator-() const {
    LowRank A(*this);
    A.U = -U;
    A.S = -S;
    A.V = -V;
    return A;
  }

  Block LowRank::operator+(const Node& B) const {
    Block Out(*this);
    Out += B;
    return Out;
  }
  Block LowRank::operator+(Block&& B) const {
    return *this + *B.ptr;
  }
  const Node& LowRank::operator+=(const Node& B) {
    if (B.is(HICMA_LOWRANK)) {
      const LowRank& BR = static_cast<const LowRank&>(B);
      assert(dim[0]==BR.dim[0] && dim[1]==BR.dim[1]);
      if (rank+BR.rank >= dim[0]) {
        *this = LowRank(this->dense() + BR.dense(), rank);
      }
      else {
        mergeU(*this, BR);
        mergeS(*this, BR);
        mergeV(*this, BR);
      }
      return *this;
    } else if(B.is(HICMA_DENSE)) {
      const Dense& BR = static_cast<const Dense&>(B);
      assert(dim[0]==BR.dim[0] && dim[1]==BR.dim[1]);
      return this->dense() += BR;
    } else {
      std::cout << this->is_string() << " + " << B.is_string();
      std::cout << " is undefined!" << std::endl;
      return *this;
    }
  }
  const Node& LowRank::operator+=(Block&& B) {
    return *this += *B.ptr;
  }

  Block LowRank::operator-(const Node& B) const {
    Block Out(*this);
    Out -= B;
    return Out;
  }
  Block LowRank::operator-(Block&& B) const {
    return *this - *B.ptr;
  }
  const Node& LowRank::operator-=(const Node& B) {
    if (B.is(HICMA_LOWRANK)) {
      const LowRank& BR = static_cast<const LowRank&>(B);
      assert(dim[0]==BR.dim[0] && dim[1]==BR.dim[1]);
      if (rank+BR.rank >= dim[0]) {
        *this = LowRank(this->dense() - BR.dense(), rank);
      }
      else {
        mergeU(*this, -BR);
        mergeS(*this, -BR);
        mergeV(*this, -BR);
      }
      return *this;
    } else if(B.is(HICMA_DENSE)) {
      const Dense& BR = static_cast<const Dense&>(B);
      assert(dim[0]==BR.dim[0] && dim[1]==BR.dim[1]);
      return this->dense() -= BR;
    } else {
      std::cout << this->is_string() << " + " << B.is_string();
      std::cout << " is undefined!" << std::endl;
      return *this;
    }
  }
  const Node& LowRank::operator-=(Block&& B) {
    return *this -= *B.ptr;
  }

  Block LowRank::operator*(const Node& B) const {
    if (B.is(HICMA_LOWRANK)) {
      const LowRank& BR = static_cast<const LowRank&>(B);
      assert(dim[1] == BR.dim[0]);
      LowRank Out(dim[0], BR.dim[1], rank);
      Out.U = U;
      Out.S = S * (V * BR.U) * BR.S;
      Out.V = BR.V;
      return Out;
    } else if(B.is(HICMA_DENSE)) {
      const Dense& BR = static_cast<const Dense&>(B);
      assert(dim[1] == BR.dim[0]);
      LowRank Out(dim[0], BR.dim[1], rank);
      Out.U = U;
      Out.S = S;
      Out.V = V * B;
      return Out;
    } else {
      std::cout << this->is_string() << " * " << B.is_string();
      std::cout << " is undefined!" << std::endl;
      return Block();
    }
  }
  Block LowRank::operator*(Block&& B) const {
    return *this * *B.ptr;
  }

  const bool LowRank::is(const int enum_id) const {
    return enum_id == HICMA_LOWRANK;
  }

  const char* LowRank::is_string() const { return "LowRank"; }

  void LowRank::resize(int m, int n, int k) {
    dim[0]=m; dim[1]=n; rank=k;
    U.resize(m,k);
    S.resize(k,k);
    V.resize(k,n);
  }

  Dense LowRank::dense() const {
    return U * S * V;
  }

  double LowRank::norm() const {
    return this->dense().norm();
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
        *this -= A * B;
      } else if (B.is(HICMA_HIERARCHICAL)) {
        fprintf(
            stderr,"%s += %s * %s undefined.\n",
            this->is_string(), A.is_string(), B.is_string());
        abort();
      }
    } else if (A.is(HICMA_LOWRANK)) {
      if (B.is(HICMA_DENSE)) {
        *this -= A * B;
      } else if (B.is(HICMA_LOWRANK)) {
        *this -= A * B;
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
