#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))
#define RAND_MAX 1

void initialize_random_matrix(double *M, int nrows, int ncols){
  time_t t;
  srand((unsigned) time(&t));
  for(int i=0; i<nrows; i++){
    for(int j=0; j<ncols; j++){
      M[i*ncols + j] = rand();
    }
  }
}

/* C = A^T*B */
void matrix_transpose_matrix_mult(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C){
    gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}

/* compute compact QR factorization
M is mxn; Q is mxk and R is kxk
*/
void compute_QR_compact_factorization(gsl_matrix *M, gsl_matrix *Q, gsl_matrix *R){
  int m = M->size1;
  int n = M->size2;
  int k = min(m,n);
  gsl_matrix *QR = gsl_matrix_calloc(M->size1, M->size2);
  gsl_vector *tau = gsl_vector_alloc(min(M->size1,M->size2));
  gsl_matrix_memcpy (QR, M);
  gsl_linalg_QR_decomp (QR, tau);
  for(int i=0; i<k; i++){
    for(int j=0; j<k; j++){
      if(j>=i){
        gsl_matrix_set(R,i,j,gsl_matrix_get(QR,i,j));
      }
    }
  }
  gsl_vector *vj = gsl_vector_calloc(m);
  for(int j=0; j<k; j++){
    gsl_vector_set(vj,j,1.0);
    gsl_linalg_QR_Qvec (QR, tau, vj);
    gsl_matrix_set_col(Q,j,vj);
    vj = gsl_vector_calloc(m);
  }
}

/* compute compact QR factorization and get Q
M is mxn; Q is mxk and R is kxk (not computed)
*/
void QR_factorization_getQ(double *M, double *Q, int nrows, int ncols){
  int k = min(nrows,ncols);
  double *QR = calloc(nrows*ncols);
  double *tau = calloc(k);
  memcpy (QR, M, nrows*ncols);
  gsl_linalg_QR_decomp (QR, tau);
  gsl_vector *vj = gsl_vector_calloc(m);
  for(int j=0; j<k; j++){
    gsl_vector_set(vj,j,1.0);
    gsl_linalg_QR_Qvec (QR, tau, vj);
    gsl_matrix_set_col(Q,j,vj);
    vj = gsl_vector_calloc(m);
  }
  gsl_vector_free(vj);
  gsl_vector_free(tau);
  gsl_matrix_free(QR);
}

/* build diagonal matrix from vector elements */
void build_diagonal_matrix(gsl_vector *dvals, int n, gsl_matrix *D){
  for(int i=0; i<n; i++){
    gsl_matrix_set(D,i,i,gsl_vector_get(dvals,i));
  }
}

/* frobenius norm */
double matrix_frobenius_norm(gsl_matrix *M){
  double norm = 0;
  for(int i=0; i<M->size1; i++){
    for(int j=0; j<M->size2; j++){
      double val = gsl_matrix_get(M, i, j);
      norm += val*val;
    }
  }
  norm = sqrt(norm);
  return norm;
}

/* P = U*S*V^T */
void form_svd_product_matrix(gsl_matrix *U, gsl_matrix *S, gsl_matrix *V, gsl_matrix *P){
  int n = P->size2;
  int k = S->size1;
  gsl_matrix * SVt = gsl_matrix_alloc(k,n);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, S, V, 0.0, SVt);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, U, SVt, 0.0, P);
}

/* calculate percent error between A and B: 100*norm(A - B)/norm(A) */
double get_percent_error_between_two_mats(gsl_matrix *A, gsl_matrix *B){
  int m = A->size1;
  int n = A->size2;
  gsl_matrix *A_minus_B = gsl_matrix_alloc(m,n);
  gsl_matrix_memcpy(A_minus_B, A);
  gsl_matrix_sub(A_minus_B,B);
  double normA = matrix_frobenius_norm(A);
  double normA_minus_B = matrix_frobenius_norm(A_minus_B);
  return 100.0*normA_minus_B/normA;
}

/* C = A*B */
void matrix_matrix_mult(doubl *A, double *B, double *C, int nrows, int ncols){
  int c_n = 'n'; double m1 = 1; double p1 = 1;
  dgemm_(&c_n, &c_n, &nrows, &ncols, &nrows, &m1, A, &nrows, B, &ncols, &p1, C, &nrows);
}

/* computes the approximate low rank SVD of rank k of matrix M using QR method */
void randomized_low_rank_svd2(
                              double *M,
                              int k,
                              double *U,
                              double *S,
                              double *V,
                              int nrows ,
                              int ncols)
{
  // setup mats
  U = calloc(nrows*k);
  S = calloc(k*k);
  V = calloc(ncols*k);
  // build random matrix
  double *RN = calloc(nrows*k); // calloc sets all elements to zero
  //RN = matrix_load_from_file("data/R.mtx");
  initialize_random_matrix(RN, nrows, k);
  // multiply to get matrix of random samples Y
  double *Y = malloc(nrows*k);
  matrix_matrix_mult(M, RN, Y, nrows, k);
  // build Q from Y
  double *Q = alloc(nrows*k);
  QR_factorization_getQ(Y, Q, nrows, k);
  // form Bt = Mt*Q : nxm * mxk = nxk
  gsl_matrix *Bt = gsl_matrix_alloc(n,k);
  matrix_transpose_matrix_mult(M,Q,Bt);
  gsl_matrix *Qhat = gsl_matrix_calloc(n,k);
  gsl_matrix *Rhat = gsl_matrix_calloc(k,k);
  compute_QR_compact_factorization(Bt,Qhat,Rhat);
  // compute SVD of Rhat (kxk)
  gsl_matrix *Uhat = gsl_matrix_alloc(k,k);
  gsl_vector *Sigmahat = gsl_vector_alloc(k);
  gsl_matrix *Vhat = gsl_matrix_alloc(k,k);
  gsl_vector *svd_work_vec = gsl_vector_alloc(k);
  gsl_matrix_memcpy(Uhat, Rhat);
  gsl_linalg_SV_decomp (Uhat, Vhat, Sigmahat, svd_work_vec);
  // record singular values
  build_diagonal_matrix(Sigmahat, k, *S);
  // U = Q*Vhat
  matrix_matrix_mult(Q,Vhat,*U);
  // V = Qhat*Uhat
  matrix_matrix_mult(Qhat,Uhat,*V);
  // free stuff
  gsl_matrix_free(RN);
  gsl_matrix_free(Y);
  gsl_matrix_free(Q);
  gsl_matrix_free(Rhat);
  gsl_matrix_free(Qhat);
  gsl_matrix_free(Uhat);
  gsl_matrix_free(Vhat);
  gsl_matrix_free(Bt);
}
