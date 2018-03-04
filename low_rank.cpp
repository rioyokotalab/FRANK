#include "low_rank.h"

namespace hicma {
  LowRank::LowRank(const LowRank &A) : U(A.U), B(A.B), V(A.V) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
  }

  LowRank::LowRank(const Dense &D, const int k) {
    int m = dim[0] = D.dim[0];
    int n = dim[1] = D.dim[1];
    rank = k;
    U.resize(m,k);
    B.resize(k,k);
    V.resize(k,n);
    gsl_matrix *D2 = gsl_matrix_calloc(m,n);
    gsl_matrix *U2 = gsl_matrix_calloc(m,k);
    gsl_matrix *B2 = gsl_matrix_calloc(k,k);
    gsl_matrix *V2 = gsl_matrix_calloc(n,k);
    for(int i=0; i<m; i++){
      for(int j=0; j<n; j++){
        D2->data[i*n+j] = D(i,j);
      }
    }
    gsl_matrix *RN = gsl_matrix_calloc(n,k);
    initialize_random_matrix(RN);
    gsl_matrix *Y = gsl_matrix_alloc(m,k);
    matrix_matrix_mult(D2, RN, Y);
    gsl_matrix *Q = gsl_matrix_alloc(m,k);
    QR_factorization_getQ(Y, Q);
    gsl_matrix *Bt = gsl_matrix_alloc(n,k);
    matrix_transpose_matrix_mult(D2,Q,Bt);
    gsl_matrix *Qhat = gsl_matrix_calloc(n,k);
    gsl_matrix *Rhat = gsl_matrix_calloc(k,k);
    compute_QR_compact_factorization(Bt,Qhat,Rhat);
    gsl_matrix *Uhat = gsl_matrix_alloc(k,k);
    gsl_vector *Sigmahat = gsl_vector_alloc(k);
    gsl_matrix *Vhat = gsl_matrix_alloc(k,k);
    gsl_vector *svd_work_vec = gsl_vector_alloc(k);
    gsl_matrix_memcpy(Uhat, Rhat);
    gsl_linalg_SV_decomp (Uhat, Vhat, Sigmahat, svd_work_vec);
    build_diagonal_matrix(Sigmahat, k, B2);
    matrix_matrix_mult(Q,Vhat,U2);
    matrix_matrix_mult(Qhat,Uhat,V2);
    for(int i=0; i<m; i++){
      for(int j=0; j<k; j++){
        U(i,j) = U2->data[i*k+j];
      }
    }
    for(int i=0; i<k; i++){
      for(int j=0; j<k; j++){
        B(i,j) = B2->data[i*k+j];
      }
    }
    for(int i=0; i<n; i++){
      for(int j=0; j<k; j++){
        V(j,i) = V2->data[i*k+j];
      }
    }
    gsl_matrix_free(D2);
    gsl_matrix_free(U2);
    gsl_matrix_free(B2);
    gsl_matrix_free(V2);
    gsl_matrix_free(Y);
    gsl_matrix_free(Q);
    gsl_matrix_free(Rhat);
    gsl_matrix_free(Qhat);
    gsl_matrix_free(Uhat);
    gsl_matrix_free(Vhat);
    gsl_matrix_free(Bt);
  }

  const LowRank& LowRank::operator=(const LowRank A) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
    U = A.U;
    B = A.B;
    V = A.V;
    return *this;
  }

  Dense LowRank::operator+(const Dense& D) const {
    return U * B * V + D;
  }

  /*
    LowRank LowRank::operator*(const Dense& D) {
    V = V * D;
    return *this;
    }
  */

  /*
    LowRank LowRank::operator*(const LowRank& A) {
    B = B * (V * A.U) * A.B;
    return *this;
    }
  */

  Dense LowRank::dense() {
    return (U * B * V);
  }
}
