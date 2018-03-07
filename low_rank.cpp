#include "dense.h"
#include "low_rank.h"
#include "hierarchical.h"

namespace hicma {
  LowRank::LowRank() {
    dim[0]=0; dim[1]=0; rank=0;
  }

  LowRank::LowRank(const int m, const int n, const int k) {
    dim[0]=m; dim[1]=n; rank=k;
    U.resize(m,k);
    S.resize(k,k);
    V.resize(k,n);
  }

  LowRank::LowRank(const LowRank &A) : U(A.U), S(A.S), V(A.V) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
  }

  LowRank::LowRank(const Dense &A, const int k) {
    int m = dim[0] = A.dim[0];
    int n = dim[1] = A.dim[1];
    rank = k;
    U.resize(m,k);
    S.resize(k,k);
    V.resize(k,n);
    double *A2 = (double*)calloc(m*n, sizeof(double));
    double *U2 = (double*)calloc(m*k, sizeof(double));
    double *S2 = (double*)calloc(k*k, sizeof(double));
    double *V2 = (double*)calloc(n*k, sizeof(double));
    double *V2_t = (double*)calloc(k*n, sizeof(double));
    for(int i=0; i<m; i++){
      for(int j=0; j<n; j++){
        A2[i*n+j] = A[i*n+j];
      }
    }
    randomized_low_rank_svd2(A2, rank, U2, S2, V2, m, n);
    transpose(V2, V2_t, n, k);

    for(int i=0; i<m; i++){
      for(int j=0; j<k; j++){
        U(i,j) = U2[i*k+j];
      }
    }
    for(int i=0; i<k; i++){
      for(int j=0; j<k; j++){
        S(i,j) = S2[i*k+j];
      }
    }
    for(int i=0; i<n; i++){
      for(int j=0; j<k; j++){
        V(j,i) = V2[i*k+j];
      }
    }
    free(A2);
    free(U2);
    free(S2);
    free(V2);
  }

  const LowRank& LowRank::operator=(const double v) {
    assert(v == 0);
    U = 0;
    S = 0;
    V = 0;
    return *this;
  }

  const LowRank& LowRank::operator=(const LowRank A) {
    assert(dim[0]==A.dim[0] && dim[1]==A.dim[1] && rank==A.rank);
    dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
    U = A.U;
    S = A.S;
    V = A.V;
    return *this;
  }

  const Dense LowRank::operator+=(const Dense& A) {
    assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
    return this->dense() + A;
  }

  const LowRank LowRank::operator+=(const LowRank& B) {
    assert(dim[0]==B.dim[0] && dim[1]==B.dim[1]);
    LowRank A(*this);
    if (rank+B.rank >= dim[0]) {
      *this = LowRank(A.dense() + B.dense(), rank);
    }
    else {
      this->resize(dim[0], dim[1], rank+A.rank);
      this->mergeU(A,B);
      this->mergeS(A,B);
      this->mergeV(A,B);
    }
    return *this;
  }

  const Dense LowRank::operator-=(const Dense& A) {
    assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
    return this->dense() - A;
  }

  const LowRank LowRank::operator-=(const LowRank& B) {
    assert(dim[0]==B.dim[0] && dim[1]==B.dim[1]);
    LowRank A(*this);
    if (rank+B.rank >= dim[0]) {
      *this = LowRank(A.dense() - B.dense(), rank);
    }
    else {
      this->resize(dim[0], dim[1], rank+A.rank);
      this->mergeU(A,-B);
      this->mergeS(A,-B);
      this->mergeV(A,-B);
    }
    return *this;
  }

  const LowRank LowRank::operator*=(const Dense& A) {
    LowRank B(dim[0],A.dim[1],rank);
    B.U = U;
    B.S = S;
    B.V = V * A;
    return B;
  }

  const LowRank LowRank::operator*=(const LowRank& A) {
    LowRank B(dim[0],A.dim[1],rank);
    B.U = U;
    B.S = S * (V * A.U) * A.S;
    B.V = A.V;
    return B;
  }

  Dense LowRank::operator+(const Dense& A) const {
    return LowRank(*this) += A;
  }

  LowRank LowRank::operator+(const LowRank& A) const {
    return LowRank(*this) += A;
  }

  Dense LowRank::operator-(const Dense& A) const {
    return LowRank(*this) -= A;
  }

  LowRank LowRank::operator-(const LowRank& A) const {
    return LowRank(*this) -= A;
  }

  LowRank LowRank::operator*(const Dense& A) const {
    return LowRank(*this) *= A;
  }

  LowRank LowRank::operator*(const LowRank& A) const {
    return LowRank(*this) *= A;
  }

  LowRank LowRank::operator-() const {
    LowRank A(*this);
    A.U = -U;
    A.S = -S;
    A.V = -V;
    return A;
  }

  void LowRank::resize(int m, int n, int k) {
    dim[0]=m; dim[1]=n; rank=k;
    U.resize(m,k);
    S.resize(k,k);
    V.resize(k,n);
  }

  Dense LowRank::dense() const {
    return (U * S * V);
  }

  void LowRank::mergeU(const LowRank&A, const LowRank& B) {
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

  void LowRank::mergeS(const LowRank&A, const LowRank& B) {
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

  void LowRank::mergeV(const LowRank&A, const LowRank& B) {
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

  void LowRank::trsm(Dense& A, const char& uplo) {
    switch (uplo) {
    case 'l' :
      U.trsm(A, uplo);
      break;
    case 'u' :
      V.trsm(A, uplo);
      break;
    }
  }

  void LowRank::gemm(const LowRank& A, const Dense& B) {
    Dense D = this->dense();
    D.gemm(A, B);
    *this = LowRank(D, this->rank);
  }

  void LowRank::gemm(const Dense& A, const LowRank& B) {
    Dense D = this->dense();
    D.gemm(A, B);
    *this = LowRank(D, this->rank);
  }

  void LowRank::gemm(const LowRank& A, const LowRank& B) {
    Dense D = this->dense();
    D.gemm(A, B);
    *this = LowRank(D, this->rank);
  }
}
