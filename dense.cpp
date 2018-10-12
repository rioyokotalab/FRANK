#include "hierarchical.h"

#include <lapacke.h>

namespace hicma {

  Dense::Dense() {
    dim[0]=0; dim[1]=0;
  }

  Dense::Dense(const int m) {
    dim[0]=m; dim[1]=1; data.resize(dim[0],0);
  }

  Dense::Dense(
               const int m,
               const int n,
               const int i_abs,
               const int j_abs,
               const int level
               ) : Node(i_abs, j_abs, level) {
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
               const int i_abs,
               const int j_abs,
               const int level
               ) : Node(i_abs,j_abs,level) {
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

  Dense::Dense(const LowRank& A) : Node(A.i_abs,A.j_abs,A.level) {
    Dense UxS(A.dim[0], A.rank);
    UxS.gemm(A.U, A.S);
    Dense UxSxV(A.dim[0], A.dim[1], A.i_abs, A.j_abs, A.level);
    UxSxV.gemm(UxS, A.V);
    *this = UxSxV;
  }

  Dense::Dense(const Hierarchical& A) : Node(A.i_abs,A.j_abs,A.level) {
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
            D(ic+i_begin, jc+j_begin) = AD(ic,jc);
          }
        }
        j_begin += AD.dim[1];
      }
      i_begin += AA.dim[0];
    }
    data = D.data;
  }

  Dense::Dense(const Node& _A) : Node(_A.i_abs, _A.j_abs, _A.level) {
    if (_A.is(HICMA_DENSE)) {
      *this = static_cast<const Dense&>(_A);
    } else if (_A.is(HICMA_LOWRANK)) {
      *this = Dense(static_cast<const LowRank&>(_A));
    } else if (_A.is(HICMA_HIERARCHICAL)) {
      *this = Dense(static_cast<const Hierarchical&>(_A));
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

  const Dense& Dense::operator=(const double a) {
    for (int i=0; i<dim[0]*dim[1]; i++)
      data[i] = a;
    return *this;
  }

  const Dense& Dense::operator=(Dense A) {
    swap(*this, A);
    return *this;
  }

  Dense Dense::operator+(const Dense& A) const {
    Dense B(*this);
    B += A;
    return B;
  }

  Dense Dense::operator-(const Dense& A) const {
    Dense B(*this);
    B -= A;
    return B;
  }

  const Dense& Dense::operator+=(const Dense& A) {
    assert(dim[0] == A.dim[0] && dim[1] == A.dim[1]);
    for (int i=0; i<dim[0]*dim[1]; i++) {
      (*this)[i] += A[i];
    }
    return *this;
  }

  const Dense& Dense::operator-=(const Dense& A) {
    assert(dim[0] == A.dim[0] && dim[1] == A.dim[1]);
    for (int i=0; i<dim[0]*dim[1]; i++) {
      (*this)[i] -= A[i];
    }
    return *this;
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

  bool Dense::is(const int enum_id) const {
    return enum_id == HICMA_DENSE;
  }

  const char* Dense::type() const { return "Dense"; }

  double Dense::norm() const {
    double l2 = 0;
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        l2 += data[i*dim[1]+j] * data[i*dim[1]+j];
      }
    }
    return l2;
  }

  void Dense::resize(const int dim0, const int dim1) {
    for (int i=0; i<dim0; i++) {
      for (int j=0; j<dim1; j++) {
        data[i*dim1+j] = data[i*dim[1]+j];
      }
    }
    dim[0] = dim0;
    dim[1] = dim1;
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

  void Dense::gemm(const Dense& A, const Dense&B, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                   const int& alpha, const int& beta) {
    int k = TransA == CblasNoTrans ? A.dim[1] : A.dim[0];
    cblas_dgemm(CblasRowMajor, TransA, TransB, dim[0], dim[1], k, alpha,
                &A[0], A.dim[1], &B[0], B.dim[1], beta, &data[0], dim[1]);
  }

  void Dense::gemm(const Node& _A, const Node& _B, const int& alpha, const int& beta) {
    if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      assert(this->dim[0] == A.dim[0]);
      if (_B.is(HICMA_DENSE)) {
        const Dense& B = static_cast<const Dense&>(_B);
        assert(A.dim[1] == B.dim[0]);
        assert(this->dim[1] == B.dim[1]);
        if (B.dim[1] == 1) {
          cblas_dgemv(CblasRowMajor, CblasNoTrans, A.dim[0], A.dim[1], alpha,
                      &A[0], A.dim[1], &B[0], 1, beta, &data[0], 1);
        }
        else {
          gemm(A, B, CblasNoTrans, CblasNoTrans, alpha, beta);
        }
      } else if (_B.is(HICMA_LOWRANK)) {
        const LowRank& B = static_cast<const LowRank&>(_B);
        Dense AxU(dim[0], B.rank);
        AxU.gemm(A, B.U, 1, 0);
        Dense AxUxS(dim[0], B.rank);
        AxUxS.gemm(AxU, B.S, 1, 0);
        this->gemm(AxUxS, B.V, alpha, beta);
      } else if (_B.is(HICMA_HIERARCHICAL)) {
        const Hierarchical& B = static_cast<const Hierarchical&>(_B);
        Hierarchical C(*this, B.dim[0], B.dim[1]);
        C.gemm(A, B, alpha, beta);
        *this = Dense(C);
      } else {
        std::cerr << this->type() << " -= " << _A.type();
        std::cerr << " * " << _B.type() << " is undefined." << std::endl;
        abort();
      }
    } else if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      if (_B.is(HICMA_DENSE)) {
        const Dense& B = static_cast<const Dense&>(_B);
        Dense VxB(A.rank, B.dim[1]);
        VxB.gemm(A.V, B, 1, 0);
        Dense SxVxB(A.rank, B.dim[1]);
        SxVxB.gemm(A.S, VxB, 1, 0);
        this->gemm(A.U, SxVxB, alpha, beta);
      } else if (_B.is(HICMA_LOWRANK)) {
        const LowRank& B = static_cast<const LowRank&>(_B);
        Dense VxU(A.rank, B.rank);
        VxU.gemm(A.V, B.U, 1, 0);
        Dense SxVxU(A.rank, B.rank);
        SxVxU.gemm(A.S, VxU, 1, 0);
        Dense SxVxUxS(A.rank, B.rank);
        SxVxUxS.gemm(SxVxU, B.S, 1, 0);
        Dense UxSxVxUxS(A.dim[0], B.rank);
        UxSxVxUxS.gemm(A.U, SxVxUxS, 1, 0);
        this->gemm(UxSxVxUxS, B.V, alpha, beta);
      } else if (_B.is(HICMA_HIERARCHICAL)) {
        const Hierarchical& B = static_cast<const Hierarchical&>(_B);
        Hierarchical C(*this, B.dim[0], B.dim[1]);
        C.gemm(A, B, alpha, beta);
        *this = Dense(C);
      } else {
        std::cerr << this->type() << " -= " << _A.type();
        std::cerr << " * " << _B.type() << " is undefined." << std::endl;
        abort();
      }
    } else if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      if (_B.is(HICMA_LOWRANK)) {
        const LowRank& B = static_cast<const LowRank&>(_B);
        Hierarchical C(*this, A.dim[0], A.dim[1]);
        C.gemm(A, B, alpha, beta);
        *this = Dense(C);
      } else if (_B.is(HICMA_DENSE)) {
        const Dense& B = static_cast<const Dense&>(_B);
        Hierarchical C(*this, A.dim[0], A.dim[1]);
        C.gemm(A, B, alpha, beta);
        *this = Dense(C);
      } else {
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

  void Dense::qr(Dense& Q, Dense& R) {
    std::vector<double> tau(dim[1]);
    for (int i=0; i<dim[1]; i++) Q[i*dim[1]+i] = 1.0;
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, dim[0], dim[1], &data[0], dim[1], &tau[0]);
    LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'N', dim[0], dim[1], dim[1],
                   &data[0], dim[1], &tau[0], &Q[0], dim[1]);
    for(int i=0; i<dim[1]; i++) {
      for(int j=0; j<dim[1]; j++) {
        if(j>=i){
          R[i*dim[1]+j] = data[i*dim[1]+j];
        }
      }
    }
  }

  void Dense::svd(Dense& U, Dense& S, Dense& V) {
    Dense Sdiag(dim[0],1);
    Dense work(dim[1]-1,1);
    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', dim[0], dim[1], &data[0], dim[0],
                   &Sdiag[0], &U[0], dim[0], &V[0], dim[1], &work[0]);
    for(int i=0; i<dim[0]; i++){
      S[i*dim[1]+i] = Sdiag[i];
    }
  }
}
