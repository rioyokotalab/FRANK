#include "hierarchical.h"

#include <lapacke.h>
#include <cblas.h>

namespace hicma {

  Dense::Dense() {
    dim[0]=0; dim[1]=0;
  }

  Dense::Dense(const int m) {
    dim[0]=m; dim[1]=1; data.resize(dim[0],0);
  }

  Dense::Dense(const int m, const int n) {
    dim[0]=m; dim[1]=n; data.resize(dim[0]*dim[1],0);
  }

  Dense::Dense(
               void (*func)(
                            std::vector<double>& data,
                            std::vector<double>& x,
                            const int& ni,
                            const int& nj,
                            const int& i_begin,
                            const int& j_begin
                            ),
               std::vector<double>& x,
               const int ni,
               const int nj,
               const int i_begin,
               const int j_begin,
               const int _i_abs,
               const int _j_abs,
               const int _level
               ) : Node(_i_abs,_j_abs,_level) {
    dim[0] = ni; dim[1] = nj;
    data.resize(dim[0]*dim[1]);
    func(data, x, ni, nj, i_begin, j_begin);
  }

  Dense::Dense(const Dense& A) : Node(A.i_abs,A.j_abs,A.level), data(A.data) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1];
  }

  Dense::Dense(Dense&& A) {
    swap(*this, A);
  }

  Dense::Dense(const Dense* A) : Node(A->i_abs,A->j_abs,A->level), data(A->data) {
    dim[0]=A->dim[0]; dim[1]=A->dim[1];
  }

  Dense::Dense(const Block& _A) : Node((*_A.ptr).i_abs, (*_A.ptr).j_abs, (*_A.ptr).level) {
    if (_A.is(HICMA_DENSE)) {
      Dense& A = static_cast<Dense&>(*_A.ptr);
      dim[0]=A.dim[0]; dim[1]=A.dim[1];
      data = A.data;
    } else if (_A.is(HICMA_LOWRANK)) {
      LowRank& A = static_cast<LowRank&>(*_A.ptr);
      Dense AD = A.U * A.S * A.V;
      dim[0]=AD.dim[0]; dim[1]=AD.dim[1];
      data = AD.data;
    } else if (_A.is(HICMA_HIERARCHICAL)) {
      Hierarchical& A = static_cast<Hierarchical&>(*_A.ptr);
      dim[0] = 0;
      for (int i=0; i<A.dim[0]; i++) {
        Dense AD = Dense(A(i,0));
        dim[0] += AD.dim[0];
      }
      dim[1] = 0;
      for (int j=0; j<A.dim[1]; j++) {
        Dense AD = Dense(A(0,j));
        dim[1] += AD.dim[1];
      }
      Dense D(dim[0],dim[1]);
      int i_begin = 0;
      for (int i=0; i<A.dim[0]; i++) {
        Dense AA = Dense(A(i,0));
        int j_begin = 0;
        for (int j=0; j<A.dim[1]; j++) {
          Dense AD = Dense(A(i,j));
          for (int ic=0; ic<AD.dim[0]; ic++) {
            for (int jc=0; jc<AD.dim[1]; jc++) {
              D(ic+i_begin,jc+j_begin) = AD(ic,jc);
            }
          }
          j_begin += AD.dim[1];
        }
        i_begin += AA.dim[0];
      }
      data = D.data;
    }
  }

  Dense* Dense::clone() const {
    return new Dense(*this);
  }

  void swap(Dense& A, Dense& B) {
    using std::swap;
    swap(A.data, B.data);
    swap(A.dim, B.dim);
    swap(A.i_abs, B.i_abs);
    swap(A.j_abs, B.j_abs);
    swap(A.level, B.level);
  }

  const Node& Dense::operator=(const double a) {
    for (int i=0; i<dim[0]*dim[1]; i++)
      data[i] = a;
    return *this;
  }

  const Node& Dense::operator=(const Node& _A) {
    if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      dim[0] = A.dim[0]; dim[1] = A.dim[1];
      //data.resize(dim[0]*dim[1]);
      data = A.data;
      return *this;
    } else {
      std::cerr << this->type() << " = " << _A.type();
      std::cerr << " is undefined." << std::endl;
      return *this;
    }
  };

  const Node& Dense::operator=(Node&& A) {
    if (A.is(HICMA_DENSE)) {
      swap(*this, static_cast<Dense&>(A));
      return *this;
    } else {
      std::cerr << this->type() << " = " << A.type();
      std::cerr << " is undefined." << std::endl;
      return *this;
    }
  };

  const Dense& Dense::operator=(Dense A) {
    swap(*this, A);
    return *this;
  }

  const Node& Dense::operator=(Block A) {
    return *this = std::move(*A.ptr);
  }

  Dense Dense::operator-() const {
    Dense A(dim[0],dim[1]);
    for (int i=0; i<dim[0]*dim[1]; i++) A[i] = -data[i];
    return A;
  }

  Block Dense::operator+(const Node& A) const {
    Block B(*this);
    B += A;
    return B;
  }
  Block Dense::operator+(Block&& A) const {
    return *this + *A.ptr;
  }
  const Node& Dense::operator+=(const Node& _A) {
    if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      assert(dim[0] == A.dim[0] && dim[1] == A.dim[1]);
      for (int i=0; i<dim[0]*dim[1]; i++) {
        (*this)[i] += A[i];
      }
      return *this;
    } else if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      assert(dim[0] == A.dim[0] && dim[1] == A.dim[1]);
      return *this += A.dense();
    } else {
      std::cerr << this->type() << " + " << _A.type();
      std::cerr << " is undefined." << std::endl;
      return *this;
    }
  }
  const Node& Dense::operator+=(Block&& A) {
    return *this += *A.ptr;
  }

  Block Dense::operator-(const Node& _A) const {
    Block A(*this);
    A -= _A;
    return A;
  }

  Block Dense::operator-(Block&& A) const {
    return *this - *A.ptr;
  }

  const Node& Dense::operator-=(const Node& _A) {
    if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      assert(dim[0] == A.dim[0] && dim[1] == A.dim[1]);
      for (int i=0; i<dim[0]*dim[1]; i++) {
        (*this)[i] -= A[i];
      }
      return *this;
    } else if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      assert(dim[0] == A.dim[0] && dim[1] == A.dim[1]);
      return *this -= A.dense();
    } else {
      std::cerr << this->type() << " - " << _A.type();
      std::cerr << " is undefined." << std::endl;
      return *this;
    }
  }

  const Node& Dense::operator-=(Block&& A) {
    return *this -= *A.ptr;
  }

  Block Dense::operator*(const Node& _A) const {
    if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      assert(dim[0] == A.dim[0] && dim[1] == A.dim[1]);
      LowRank B(A);
      B.U = *this * A.U;
      return B;
    } else if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      assert(dim[1] == A.dim[0]);
      Dense B(dim[0], A.dim[1]);
      if (A.dim[1] == 1) {
        cblas_dgemv(
                    CblasRowMajor,
                    CblasNoTrans,
                    dim[0],
                    dim[1],
                    1,
                    &data[0],
                    dim[1],
                    &A[0],
                    1,
                    0,
                    &B[0],
                    1
                    );
      }
      else {
        cblas_dgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    B.dim[0],
                    B.dim[1],
                    dim[1],
                    1,
                    &data[0],
                    dim[1],
                    &A[0],
                    A.dim[1],
                    0,
                    &B[0],
                    B.dim[1]
                    );
      }
      return B;
    } else {
      std::cerr << this->type() << " * " << _A.type();
      std::cerr << " is undefined." << std::endl;
      return Block();
    }
  }

  Block Dense::operator*(Block&& A) const {
    return *this * *A.ptr;
  }

  double& Dense::operator[](const int i) {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  const double& Dense::operator[](const int i) const {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  double& Dense::operator()(const int i, const int j) {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  const double& Dense::operator()(const int i, const int j) const {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  const bool Dense::is(const int enum_id) const {
    return enum_id == HICMA_DENSE;
  }

  const char* Dense::type() const { return "Dense"; }

  void Dense::resize(int i) {
    dim[0]=i; dim[1]=1;
    data.resize(dim[0]*dim[1]);
  }

  void Dense::resize(int i, int j) {
    dim[0]=i; dim[1]=j;
    data.resize(dim[0]*dim[1]);
  }

  double Dense::norm() const {
    double l2 = 0;
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        l2 += data[i*dim[1]+j] * data[i*dim[1]+j];
      }
    }
    return l2;
  }

  void Dense::print() const {
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        std::cout << std::setw(20) << std::setprecision(15) << data[i*dim[1]+j] << ' ';
      }
      std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
  }

  void Dense::getrf() {
    std::vector<int> ipiv(std::min(dim[0],dim[1]));
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim[0], dim[1], &data[0], dim[1], &ipiv[0]);
  }

  void Dense::trsm(const Node& _A, const char& uplo) {
    if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      if (dim[1] == 1) {
        switch (uplo) {
        case 'l' :
          cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                      dim[0], dim[1], 1, &A[0], A.dim[1], &data[0], dim[1]);
          break;
        case 'u' :
          cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                      dim[0], dim[1], 1, &A[0], A.dim[1], &data[0], dim[1]);
          break;
        default :
          std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
          abort();
        }
      }
      else {
        switch (uplo) {
        case 'l' :
          cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                      dim[0], dim[1], 1, &A[0], A.dim[1], &data[0], dim[1]);
          break;
        case 'u' :
          cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                      dim[0], dim[1], 1, &A[0], A.dim[1], &data[0], dim[1]);
          break;
        default :
          std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
          abort();
        }
      }
    } else if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      Hierarchical H(*this, A.dim[0], A.dim[1]);
      H.trsm(A, uplo);
      *this = Dense(H);
    } else {
      std::cerr << this->type() << " /= " << _A.type();
      std::cerr << " is undefined." << std::endl;
      abort();
    }
  }

  void Dense::gemm(const Node& _A, const Node& _B) {
    if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      assert(this->dim[0] == A.dim[0]);
      if (_B.is(HICMA_DENSE)) {
        const Dense& B = static_cast<const Dense&>(_B);
        assert(A.dim[1] == B.dim[0]);
        assert(this->dim[1] == B.dim[1]);
        if (A.dim[1] == 1) {
          cblas_dgemv(
                      CblasRowMajor,
                      CblasNoTrans,
                      A.dim[0],
                      A.dim[1],
                      -1,
                      &A[0],
                      A.dim[1],
                      &B[0],
                      1,
                      1,
                      &data[0],
                      1
                      );
        }
        else {
          cblas_dgemm(
                      CblasRowMajor,
                      CblasNoTrans,
                      CblasNoTrans,
                      dim[0],
                      dim[1],
                      A.dim[1],
                      -1,
                      &A[0],
                      A.dim[1],
                      &B[0],
                      B.dim[1],
                      1,
                      &data[0],
                      dim[1]
                      );
        }
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
#if 1
        VxU.gemm(A.V,B.U);
#else
        VxU -= A.V * B.U;
#endif
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
