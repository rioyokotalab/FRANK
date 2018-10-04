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
                   ) : Node(A.ptr->i_abs,A.ptr->j_abs,A.ptr->level) {
    assert(A.is(HICMA_DENSE));
    const Dense& B = static_cast<Dense&>(*A.ptr);
    int m = dim[0] = B.dim[0];
    int n = dim[1] = B.dim[1];
    rank = k;
    U = Dense(m,k);
    S = Dense(k,k);
    V = Dense(k,n);
    randomized_low_rank_svd2(B.data, rank, U.data, S.data, V.data, m, n);
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

  const Node& LowRank::operator=(const Node& A) {
    if (A.is(HICMA_LOWRANK)) {
      const LowRank& B = static_cast<const LowRank&>(A);
      assert(dim[0]==B.dim[0] && dim[1]==B.dim[1] && rank==B.rank);
      dim[0]=B.dim[0]; dim[1]=B.dim[1]; rank=B.rank;
      U = B.U;
      S = B.S;
      V = B.V;
      return *this;
    } else {
      std::cerr << this->is_string() << " = " << A.is_string();
      std::cerr << " is undefined." << std::endl;
      abort();
      return *this;
    }
  }

  const Node& LowRank::operator=(Node&& A) {
    if (A.is(HICMA_LOWRANK)) {
      swap(*this, static_cast<LowRank&>(A));
      return *this;
    } else {
      std::cerr << this->is_string() << " = " << A.is_string();
      std::cerr << " is undefined." << std::endl;
      abort();
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

  Block LowRank::operator+(const Node& A) const {
    Block B(*this);
    B += A;
    return B;
  }

  Block LowRank::operator+(Block&& A) const {
    return *this + *A.ptr;
  }

  const Node& LowRank::operator+=(const Node& A) {
    if (A.is(HICMA_DENSE)) {
      const Dense& B = static_cast<const Dense&>(A);
      assert(dim[0]==B.dim[0] && dim[1]==B.dim[1]);
      return this->dense() += B;
    } else if (A.is(HICMA_LOWRANK)) {
      const LowRank& B = static_cast<const LowRank&>(A);
      assert(dim[0]==B.dim[0] && dim[1]==B.dim[1]);
      if (rank+B.rank >= dim[0]) {
        *this = LowRank(this->dense() + B.dense(), rank);
      } else {
        mergeU(*this, B);
        mergeS(*this, B);
        mergeV(*this, B);
      }
      return *this;
    } else {
      std::cerr << this->is_string() << " + " << A.is_string();
      std::cerr << " is undefined." << std::endl;
      abort();
      return *this;
    }
  }

  const Node& LowRank::operator+=(Block&& A) {
    return *this += *A.ptr;
  }

  Block LowRank::operator-(const Node& A) const {
    Block B(*this);
    B -= A;
    return B;
  }
  Block LowRank::operator-(Block&& A) const {
    return *this - *A.ptr;
  }
  const Node& LowRank::operator-=(const Node& A) {
    if(A.is(HICMA_DENSE)) {
      const Dense& B = static_cast<const Dense&>(A);
      assert(dim[0]==B.dim[0] && dim[1]==B.dim[1]);
      return this->dense() -= B;
    } else if (A.is(HICMA_LOWRANK)) {
      const LowRank& B = static_cast<const LowRank&>(A);
      assert(dim[0]==B.dim[0] && dim[1]==B.dim[1]);
      if (rank+B.rank >= dim[0]) {
        *this = LowRank(this->dense() - B.dense(), rank);
      } else {
        mergeU(*this, -B);
        mergeS(*this, -B);
        mergeV(*this, -B);
      }
      return *this;
    } else {
      std::cerr << this->is_string() << " + " << A.is_string();
      std::cerr << " is undefined." << std::endl;
      abort();
      return *this;
    }
  }
  const Node& LowRank::operator-=(Block&& A) {
    return *this -= *A.ptr;
  }

  Block LowRank::operator*(const Node& A) const {
    if(A.is(HICMA_DENSE)) {
      const Dense& B = static_cast<const Dense&>(A);
      assert(dim[1] == B.dim[0]);
      LowRank C(dim[0], B.dim[1], rank);
      C.U = U;
      C.S = S;
      C.V = V * A;
      return C;
    } else if (A.is(HICMA_LOWRANK)) {
      const LowRank& B = static_cast<const LowRank&>(A);
      assert(dim[1] == B.dim[0]);
      LowRank C(dim[0], B.dim[1], rank);
      C.U = U;
      C.S = S * (V * B.U) * B.S;
      C.V = B.V;
      return C;
    } else {
      std::cerr << this->is_string() << " * " << A.is_string();
      std::cerr << " is undefined." << std::endl;
      abort();
      return Block();
    }
  }
  Block LowRank::operator*(Block&& A) const {
    return *this * *A.ptr;
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
      std::cerr << this->is_string() << " /= " << A.is_string();
      std::cerr << " is undefined." << std::endl;
      abort();
    }
  }

  void LowRank::gemm(const Node& A, const Node& B) {
    if (A.is(HICMA_DENSE)) {
      if (B.is(HICMA_DENSE)) {
        std::cerr << this->is_string() << " -= " << A.is_string();
        std::cerr << " * " << B.is_string() << " is undefined." << std::endl;
        abort();
      } else if (B.is(HICMA_LOWRANK)) {
        *this -= A * B;
      } else if (B.is(HICMA_HIERARCHICAL)) {
        std::cerr << this->is_string() << " -= " << A.is_string();
        std::cerr << " * " << B.is_string() << " is undefined." << std::endl;
        abort();
      }
    } else if (A.is(HICMA_LOWRANK)) {
      if (B.is(HICMA_DENSE)) {
        *this -= A * B;
      } else if (B.is(HICMA_LOWRANK)) {
        *this -= A * B;
      } else if (B.is(HICMA_HIERARCHICAL)) {
        std::cerr << this->is_string() << " -= " << A.is_string();
        std::cerr << " * " << B.is_string() << " is undefined." << std::endl;
        abort();
      }
    } else {
      std::cerr << this->is_string() << " -= " << A.is_string();
      std::cerr << " * " << B.is_string() << " is undefined." << std::endl;
      abort();
    }
  }
}
