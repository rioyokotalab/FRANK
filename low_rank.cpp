#include "hierarchical.h"
#include "id.h"

namespace hicma {

  LowRank::LowRank() {
    dim[0]=0; dim[1]=0; rank=0;
  }

  LowRank::LowRank(
                   const int m,
                   const int n,
                   const int k,
                   const int i_abs,
                   const int j_abs,
                   const int level) {
    dim[0]=m; dim[1]=n; rank=k;
    U = Dense(m, k, i_abs, j_abs, level);
    S = Dense(k, k, i_abs, j_abs, level);
    V = Dense(k, n, i_abs, j_abs, level);
  }

  LowRank::LowRank(const Dense& A, const int k) : Node(A.i_abs,A.j_abs,A.level) {
    dim[0] = A.dim[0];
    dim[1] = A.dim[1];
    rank = k;
    U = Dense(dim[0], rank, i_abs, j_abs, level);
    S = Dense(rank, rank, i_abs, j_abs, level);
    V = Dense(rank, dim[1], i_abs, j_abs, level);
    rsvd(A, rank, U, S, V);
  }

  LowRank::LowRank(const LowRank& A) : Node(A.i_abs,A.j_abs,A.level), U(A.U), S(A.S), V(A.V) {
     dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
  }

  LowRank::LowRank(LowRank&& A) {
    swap(*this, A);
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

  const LowRank& LowRank::operator=(LowRank A) {
    swap(*this, A);
    return *this;
  }

  const LowRank& LowRank::operator+=(const LowRank& A) {
    assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
    if (rank+A.rank >= dim[0]) {
      *this = LowRank(Dense(*this) + Dense(A), rank);
    } else {
      LowRank B(dim[0], dim[1], rank+A.rank, i_abs, j_abs, level);
      B.mergeU(*this, A);
      B.mergeS(*this, A);
      B.mergeV(*this, A);
      rank += A.rank;
      swap(U, B.U);
      swap(S, B.S);
      swap(V, B.V);
    }
    return *this;
  }

  const bool LowRank::is(const int enum_id) const {
    return enum_id == HICMA_LOWRANK;
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

  void LowRank::trsm(const Node& _A, const char& uplo) {
    if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      switch (uplo) {
      case 'l' :
        U.trsm(A, uplo);
        break;
      case 'u' :
        V.trsm(A, uplo);
        break;
      }
    } else if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      switch (uplo) {
      case 'l' :
        U.trsm(A, uplo);
        break;
      case 'u' :
        V.trsm(A, uplo);
        break;
      }
    } else {
      std::cerr << this->type() << " /= " << _A.type();
      std::cerr << " is undefined." << std::endl;
      abort();
    }
  }

  void LowRank::gemm(const Node& _A, const Node& _B, const int& alpha, const int& beta) {
    if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      if (_B.is(HICMA_DENSE)) {
        std::cerr << this->type() << " -= " << _A.type();
        std::cerr << " * " << _B.type() << " is undefined." << std::endl;
	abort();
      } else if (_B.is(HICMA_LOWRANK)) {
        const LowRank& B = static_cast<const LowRank&>(_B);
        LowRank C(B);
        C.U.gemm(A, B.U, alpha, 0);
        *this += C;
      } else if (_B.is(HICMA_HIERARCHICAL)) {
        std::cerr << this->type() << " -= " << _A.type();
        std::cerr << " * " << _B.type() << " is undefined." << std::endl;
        abort();
      }
    } else if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      if (_B.is(HICMA_DENSE)) {
        const Dense& B = static_cast<const Dense&>(_B);
        LowRank C(A);
        C.V.gemm(A.V, B, alpha, 0);
        *this += C;
      } else if (_B.is(HICMA_LOWRANK)) {
        const LowRank& B = static_cast<const LowRank&>(_B);
        LowRank C(A);
        C.V = B.V;
        Dense VxU(A.rank, B.rank);
        VxU.gemm(A.V, B.U, 1, 0);
        Dense SxVxU(A.rank, B.rank);
        SxVxU.gemm(A.S, VxU, 1, 0);
        Dense SxVxUxS(A.rank, B.rank);
        C.S.gemm(SxVxU, B.S, alpha, 0);
        *this += C;
      } else if (_B.is(HICMA_HIERARCHICAL)) {
        std::cerr << this->type() << " -= " << _A.type();
        std::cerr << " * " << _B.type() << " is undefined." << std::endl;
        abort();
      }
    } else {
      std::cerr << this->type() << " -= " << _A.type();
      std::cerr << " * " << _B.type() << " is undefined." << std::endl;
      abort();
    }
  }
}
