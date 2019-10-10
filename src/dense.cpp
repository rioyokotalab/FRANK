#include "hicma/any.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/operations.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <cassert>
#include <iostream>
#include <iomanip>

#ifndef USE_MKL
#include <lapacke.h>
#endif

#include "yorel/multi_methods.hpp"

namespace hicma {

  Dense::Dense() {
    MM_INIT();
    dim[0]=0; dim[1]=0;
  }

  Dense::Dense(const int m) {
    MM_INIT();
    dim[0]=m; dim[1]=1; data.resize(dim[0],0);
  }

  Dense::Dense(
               const int m,
               const int n,
               const int i_abs,
               const int j_abs,
               const int level
               ) : Node(i_abs, j_abs, level) {
    MM_INIT();
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
    MM_INIT();
    dim[0] = ni; dim[1] = nj;
    data.resize(dim[0]*dim[1]);
    func(data, x, ni, nj, i_begin, j_begin);
  }

  Dense::Dense(const Dense& A) : Node(A.i_abs,A.j_abs,A.level), data(A.data) {
    MM_INIT();
    dim[0]=A.dim[0]; dim[1]=A.dim[1];
  }

  Dense::Dense(Dense&& A) {
    MM_INIT();
    swap(*this, A);
  }

  Dense::Dense(const LowRank& A) : Node(A.i_abs,A.j_abs,A.level) {
    MM_INIT();
    dim[0] = A.dim[0]; dim[1] = A.dim[1];
    data.resize(dim[0]*dim[1], 0);
    Dense UxS(A.dim[0], A.rank);
    gemm(A.U, A.S, UxS, 1, 0);
    gemm(UxS, A.V, *this, 1, 0);
  }

  Dense::Dense(const Hierarchical& A) : Node(A.i_abs,A.j_abs,A.level) {
    MM_INIT();
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
    data.resize(dim[0]*dim[1]);
    int i_begin = 0;
    for (int i=0; i<A.dim[0]; i++) {
      Dense AA = Dense(A(i,0));
      int j_begin = 0;
      for (int j=0; j<A.dim[1]; j++) {
        Dense AD = Dense(A(i,j));
        for (int ic=0; ic<AD.dim[0]; ic++) {
          for (int jc=0; jc<AD.dim[1]; jc++) {
            (*this)(ic+i_begin, jc+j_begin) = AD(ic,jc);
          }
        }
        j_begin += AD.dim[1];
      }
      i_begin += AA.dim[0];
    }
  }

  Dense::Dense(const Any& A) : Node(A.ptr->i_abs, A.ptr->j_abs, A.ptr->level) {
    MM_INIT();
    *this = make_dense(*A.ptr);
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

  const Dense& Dense::operator*=(const double a) {
    for (int i=0; i<dim[0]*dim[1]; i++) {
      (*this)[i] *= a;
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
        l2 += (*this)(i, j) * (*this)(i, j);
      }
    }
    return l2;
  }

  int Dense::size() const {
    return dim[0] * dim[1];
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

  Dense Dense::transpose() const {
    Dense A(dim[1], dim[0], i_abs, j_abs, level);
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        A(j,i) = (*this)(i,j);
      }
    }
    return A;
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

  void Dense::transpose() {
    std::vector<double> _data(data);
    std::swap(dim[0], dim[1]);
    for(int i=0; i<dim[0]; i++) {
      for(int j=0; j<dim[1]; j++) {
        data[i*dim[1]+j] = _data[j*dim[0]+i];
      }
    }
  }

  void Dense::qr(Dense& Q, Dense& R) {
    start("-DGEQRF");
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
    stop("-DGEQRF",false);
  }

  void Dense::svd(Dense& U, Dense& S, Dense& V) {
    start("-DGESVD");
    Dense Sdiag(dim[0],1);
    Dense work(dim[1]-1,1);
    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', dim[0], dim[1], &data[0], dim[0],
                   &Sdiag[0], &U[0], dim[0], &V[0], dim[1], &work[0]);
    for(int i=0; i<dim[0]; i++){
      S[i*dim[1]+i] = Sdiag[i];
    }
    stop("-DGESVD",false);
  }

  void Dense::svd(Dense& Sdiag) {
    start("-DGESVD");
    Dense U(dim[0],dim[1]),V(dim[1],dim[0]);
    Dense work(dim[1]-1,1);
    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', dim[0], dim[1], &data[0], dim[0],
                   &Sdiag[0], &U[0], dim[0], &V[0], dim[1], &work[0]);
    stop("-DGESVD",false);
  }

  void Dense::geqrt(Dense& T) {
    assert(T.dim[0] == dim[1] && T.dim[1] == dim[1]);
    LAPACKE_dgeqrt3(LAPACK_ROW_MAJOR, dim[0], dim[1], &data[0], dim[1], &T[0], T.dim[1]);
  }

  void Dense::geqrt2(Dense& T) {
    assert(T.dim[0] == dim[1] && T.dim[1] == dim[1]);
    LAPACKE_dgeqrt2(LAPACK_ROW_MAJOR, dim[0], dim[1], &data[0], dim[1], &T[0], T.dim[1]);
  }

  void Dense::larfb(const Dense& Y, const Dense& T, const bool trans) {
    LAPACKE_dlarfb(LAPACK_ROW_MAJOR, 'L', (trans ? 'T' : 'N'), 'F', 'C', dim[0], dim[1], T.dim[1], &Y[0], Y.dim[1], &T[0], T.dim[1], &data[0], dim[1]);
  }

  void Dense::larfb(const Hierarchical& Y, const Hierarchical& T, const bool trans) {
    Hierarchical C(*this, Y.dim[0], Y.dim[1]);
    C.larfb(Y, T, trans);
    *this = Dense(C);
  }

  void Dense::tpqrt(Dense& A, Dense& T) {
    LAPACKE_dtpqrt2(LAPACK_ROW_MAJOR, dim[0], dim[1], 0, &A[0], A.dim[1], &data[0], dim[1], &T[0], T.dim[1]);
  }

  void Dense::tpqrt(Hierarchical& A, Dense& T) {
    print_undefined(__func__, A.type(), T.type(), this->type());
    abort();
  }

  void Dense::tpqrt(Hierarchical& A, Hierarchical& T) {
    print_undefined(__func__, A.type(), T.type(), this->type());
    abort();
  }

  void Dense::tpmqrt(Dense& B, const Dense& Y, const Dense &T, const bool trans) {
    LAPACKE_dtprfb(LAPACK_ROW_MAJOR, 'L', (trans ? 'T': 'N'), 'F', 'C', dim[0], dim[1], Y.dim[1], 0, &Y[0], Y.dim[1], &T[0], T.dim[1], &B[0], B.dim[1], &data[0], dim[1]);
  }

  void Dense::tpmqrt(Dense& B, const LowRank& Y, const Dense& T, const bool trans) {
    Dense UY(Y.U.dim[0], Y.V.dim[1]);
    gemm(Y.U, Y.V, UY, 1, 0);
    tpmqrt(B, UY, T, trans);
  }

  void Dense::tpmqrt(LowRank& B, const Dense& Y, const Dense& T, const bool trans) {
    Dense C(B);
    tpmqrt(C, Y, T, trans);
    B = LowRank(C, B.rank);
  }

  void Dense::tpmqrt(LowRank& B, const LowRank& Y, const Dense& T, const bool trans) {
    Dense C(B);
    Dense UY(Y.U.dim[0], Y.V.dim[1]);
    gemm(Y.U, Y.V, UY, 1, 0);
    tpmqrt(C, UY, T, trans);
    B = LowRank(C, B.rank);
  }

  void Dense::tpmqrt(Hierarchical& B, const Dense& Y, const Dense& T, const bool trans) {
    Hierarchical C(B);
    Dense Yt(Y);
    Yt.transpose();
    gemm(Yt, *this, C, 1, 1); // C = B + Yt.A
    Dense Tt(T);
    if(trans) Tt.transpose();
    gemm(Tt, C, B, -1, 1); // B = B - (T or Tt)*C
    Dense YTt(Y.dim[0], Tt.dim[1]);
    gemm(Y, Tt, YTt, 1, 0);
    gemm(YTt, C, *this, -1, 1); // A = A - Y*(T or Tt)*C
  }

  void Dense::tpmqrt(Hierarchical& B, const Hierarchical& Y, const Hierarchical& T, const bool trans) {
    Hierarchical A(*this, B.dim[0], B.dim[1]);
    A.tpmqrt(B, Y, T, trans);
    *this = Dense(A);
  }

  BEGIN_SPECIALIZATION(make_dense, Dense, const Hierarchical& A){
    return Dense(A);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const LowRank& A){
    return Dense(A);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const Dense& A){
    return Dense(A);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const Node& A){
    std::cout << "Cannot create dense from " << A.type() << "!" << std::endl;
    abort();
  } END_SPECIALIZATION;

}
