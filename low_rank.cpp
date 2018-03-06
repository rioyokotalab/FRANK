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

  LowRank::LowRank(const Dense &D, const int k) {
    int m = dim[0] = D.dim[0];
    int n = dim[1] = D.dim[1];
    rank = k;
    U.resize(m,k);
    S.resize(k,k);
    V.resize(k,n);
    double *D2 = (double*)calloc(m*n, sizeof(double));
    double *U2 = (double*)calloc(m*k, sizeof(double));
    double *S2 = (double*)calloc(k*k, sizeof(double));
    double *V2 = (double*)calloc(n*k, sizeof(double));
    double *V2_t = (double*)calloc(k*n, sizeof(double));
    for(int i=0; i<m; i++){
      for(int j=0; j<n; j++){
        D2[i*n+j] = D[i*n + j];
      }
    }
    randomized_low_rank_svd2(D2, rank, U2, S2, V2, m, n);
    transpose(V2, V2_t, n, k);
    // double *RN = gsl_matrix_calloc(n,k);
    // initialize_random_matrix(RN);
    // gsl_matrix *Y = gsl_matrix_alloc(m,k);
    // matrix_matrix_mult(D2, RN, Y);
    // gsl_matrix *Q = gsl_matrix_alloc(m,k);
    // QR_factorization_getQ(Y, Q);
    // gsl_matrix *Bt = gsl_matrix_alloc(n,k);
    // matrix_transpose_matrix_mult(D2,Q,Bt);
    // gsl_matrix *Qhat = gsl_matrix_calloc(n,k);
    // gsl_matrix *Rhat = gsl_matrix_calloc(k,k);
    // compute_QR_compact_factorization(Bt,Qhat,Rhat);
    // gsl_matrix *Uhat = gsl_matrix_alloc(k,k);
    // gsl_vector *Sigmahat = gsl_vector_alloc(k);
    // gsl_matrix *Vhat = gsl_matrix_alloc(k,k);
    // gsl_vector *svd_work_vec = gsl_vector_alloc(k);
    // gsl_matrix_memcpy(Uhat, Rhat);
    // gsl_linalg_SV_decomp (Uhat, Vhat, Sigmahat, svd_work_vec);
    // build_diagonal_matrix(Sigmahat, k, S2);
    // matrix_matrix_mult(Q,Vhat,U2);
    // matrix_matrix_mult(Qhat,Uhat,V2);
    
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
    free(D2);
    free(U2);
    free(S2);
    free(V2);
  }

  const LowRank& LowRank::operator=(const double v) {
    U = 0;
    S = 0;
    V = 0;
    return *this;
  }

  const LowRank& LowRank::operator=(const LowRank A) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
    U = A.U;
    S = A.S;
    V = A.V;
    return *this;
  }

  const Dense LowRank::operator+=(const Dense& D) {
    assert(dim[0]==D.dim[0] && dim[1]==D.dim[1]);
    return this->dense() + D;
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

  const Dense LowRank::operator-=(const Dense& D) {
    assert(dim[0]==D.dim[0] && dim[1]==D.dim[1]);
    return this->dense() - D;
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

  const LowRank LowRank::operator*=(const Dense& D) {
    LowRank A(dim[0],D.dim[1],rank);
    A.U = U;
    A.S = S;
    A.V = V * D;
    return A;
  }

  const LowRank LowRank::operator*=(const LowRank& A) {
    LowRank B(dim[0],A.dim[1],rank);
    B.U = U;
    B.S = S * (V * A.U) * A.S;
    B.V = A.V;
    return B;
  }

  Dense LowRank::operator+(const Dense& D) const {
    return LowRank(*this) += D;
  }

  LowRank LowRank::operator+(const LowRank& A) const {
    return LowRank(*this) += A;
  }

  Dense LowRank::operator-(const Dense& D) const {
    return LowRank(*this) -= D;
  }

  LowRank LowRank::operator-(const LowRank& A) const {
    return LowRank(*this) -= A;
  }

  LowRank LowRank::operator*(const Dense& D) const {
    return LowRank(*this) *= D;
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

  Dense LowRank::dense() const {
    return (U * S * V);
  }
}
