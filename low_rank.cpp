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
    U = Dense(m,k,i_abs,j_abs,level);
    S = Dense(k,k,i_abs,j_abs,level);
    V = Dense(k,n,i_abs,j_abs,level);
  }

  LowRank::LowRank(const Dense& A, const int k) : Node(A.i_abs,A.j_abs,A.level) {
    int m = dim[0] = A.dim[0];
    int n = dim[1] = A.dim[1];
    rank = k;
    U = Dense(m,k,i_abs,j_abs,level);
    S = Dense(k,k,i_abs,j_abs,level);
    V = Dense(k,n,i_abs,j_abs,level);
    randomized_low_rank_svd2(A.data, rank, U.data, S.data, V.data, m, n);
  }

  // TODO: This constructor is called A LOT. Work that out
  LowRank::LowRank(
                   const Block& _A,
                   const int k
                   ) : Node(_A.ptr->i_abs,_A.ptr->j_abs,_A.ptr->level) {
    assert(_A.is(HICMA_DENSE));
    const Dense& A = static_cast<Dense&>(*_A.ptr);
    int m = dim[0] = A.dim[0];
    int n = dim[1] = A.dim[1];
    rank = k;
    U = Dense(m,k,i_abs,j_abs,level);
    S = Dense(k,k,i_abs,j_abs,level);
    V = Dense(k,n,i_abs,j_abs,level);
    randomized_low_rank_svd2(A.data, rank, U.data, S.data, V.data, m, n);
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

  const Node& LowRank::operator=(const Node& _A) {
    if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      assert(dim[0]==A.dim[0] && dim[1]==A.dim[1] && rank==A.rank);
      dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
      U = A.U;
      S = A.S;
      V = A.V;
      return *this;
    } else {
      std::cerr << this->type() << " = " << _A.type();
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
      std::cerr << this->type() << " = " << A.type();
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

  Block LowRank::operator+(const Node& _A) const {
    Block A(*this);
    A += _A;
    return A;
  }

  Block LowRank::operator+(Block&& A) const {
    return *this + *A.ptr;
  }

  const Node& LowRank::operator+=(const Node& _A) {
    if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
      return this->dense() += A;
    } else if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
      if (rank+A.rank >= dim[0]) {
        *this = LowRank(this->dense() + A.dense(), rank);
      } else {
        mergeU(*this, A);
        mergeS(*this, A);
        mergeV(*this, A);
      }
      return *this;
    } else {
      std::cerr << this->type() << " + " << _A.type();
      std::cerr << " is undefined." << std::endl;
      abort();
      return *this;
    }
  }

  const Node& LowRank::operator+=(Block&& A) {
    return *this += *A.ptr;
  }

  Block LowRank::operator-(const Node& _A) const {
    Block A(*this);
    A -= _A;
    return A;
  }

  Block LowRank::operator-(Block&& A) const {
    return *this - *A.ptr;
  }

  const Node& LowRank::operator-=(const Node& _A) {
    if(_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
      return this->dense() -= A;
    } else if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
      if (rank+A.rank >= dim[0]) {
        *this = LowRank(this->dense() - A.dense(), rank);
      } else {
        mergeU(*this, -A);
        mergeS(*this, -A);
        mergeV(*this, -A);
      }
      return *this;
    } else {
      std::cerr << this->type() << " + " << _A.type();
      std::cerr << " is undefined." << std::endl;
      abort();
      return *this;
    }
  }

  const Node& LowRank::operator-=(Block&& A) {
    return *this -= *A.ptr;
  }

  Block LowRank::operator*(const Node& _A) const {
    if(_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      assert(dim[1] == A.dim[0]);
      LowRank B(dim[0], A.dim[1], rank, i_abs, j_abs, level);
      B.U = U;
      B.S = S;
      B.V = V * _A;
      return B;
    } else if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      assert(dim[1] == A.dim[0]);
      LowRank B(dim[0], A.dim[1], rank, i_abs, j_abs, level);
      B.U = U;
      B.S = S * (V * A.U) * A.S;
      B.V = A.V;
      return B;
    } else {
      std::cerr << this->type() << " * " << _A.type();
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

  const char* LowRank::type() const { return "LowRank"; }

  void LowRank::resize(int m, int n, int k) {
    dim[0]=m; dim[1]=n; rank=k;
    U.resize(m,k);
    S.resize(k,k);
    V.resize(k,n);
  }

  Dense LowRank::dense() const {
    Dense A = U * S * V;
    A.level = level;
    A.i_abs = i_abs;
    A.j_abs = j_abs;
    return std::move(A);
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

  void LowRank::gemm(const Node& _A, const Node& _B) {
    if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      if (_B.is(HICMA_DENSE)) {
        std::cerr << this->type() << " -= " << _A.type();
        std::cerr << " * " << _B.type() << " is undefined." << std::endl;
        abort();
      } else if (_B.is(HICMA_LOWRANK)) {
        const LowRank& B = static_cast<const LowRank&>(_B);
        Dense AxU(dim[0],B.rank);
        AxU.gemm(A,B.U);
        Dense AxUxS(dim[0],B.rank);
        AxUxS.gemm(AxU,B.S);
        this->gemm(AxUxS,B.V);
      } else if (_B.is(HICMA_HIERARCHICAL)) {
        std::cerr << this->type() << " -= " << _A.type();
        std::cerr << " * " << _B.type() << " is undefined." << std::endl;
        abort();
      }
    } else if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      if (_B.is(HICMA_DENSE)) {
        const Dense& B = static_cast<const Dense&>(_B);
        Dense VxB(A.rank,B.dim[1]);
        VxB.gemm(A.V,B);
        Dense SxVxB(A.rank,B.dim[1]);
        SxVxB.gemm(A.S,VxB);
        this->gemm(A.U,SxVxB);
      } else if (_B.is(HICMA_LOWRANK)) {
        const LowRank& B = static_cast<const LowRank&>(_B);
        Dense VxU(A.rank,B.rank);
        VxU.gemm(A.V,B.U);
        Dense SxVxU(A.rank,B.rank);
        SxVxU.gemm(A.S,VxU);
        Dense SxVxUxS(A.rank,B.rank);
        SxVxUxS.gemm(SxVxU,B.S);
        Dense UxSxVxUxS(A.dim[0],B.rank);
        UxSxVxUxS.gemm(A.U,SxVxUxS);
        this->gemm(UxSxVxUxS,B.V);
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
