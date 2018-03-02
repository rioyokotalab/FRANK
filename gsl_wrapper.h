#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

void initialize_random_matrix(gsl_matrix *M){
    gsl_rng_env_setup();
    const gsl_rng_type * T = gsl_rng_default;
    gsl_rng * r = gsl_rng_alloc(T);
    gsl_rng_set (r, time(NULL));
    int m = M->size1;
    int n = M->size2;
    for(int i=0; i<m; i++){
      for(int j=0; j<n; j++){
        gsl_matrix_set(M, i, j, gsl_rng_uniform (r));
      }
    }
    gsl_rng_free (r);
}

/*
% project v in direction of u
function p=project_vec(v,u)
p = (dot(v,u)/norm(u)^2)*u;
*/
void project_vector(gsl_vector *v, gsl_vector *u, gsl_vector *p){
    double dot_product_val;
    gsl_blas_ddot(v, u, &dot_product_val);
    double vec_norm = gsl_blas_dnrm2(u);
    double scalar_val = dot_product_val/(vec_norm*vec_norm);
    gsl_vector_memcpy(p, u);
    gsl_vector_scale (p, scalar_val);
}

/* C = A*B */
void matrix_matrix_mult(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C){
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}

/* C = A^T*B */
void matrix_transpose_matrix_mult(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C){
    gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}

/* compute evals and evecs of symmetric matrix */
void compute_evals_and_evecs_of_symm_matrix(gsl_matrix *M, gsl_vector *eval, gsl_matrix *evec){
    gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (M->size1);
    gsl_eigen_symmv (M, eval, evec, w);
    gsl_eigen_symmv_free(w);
    gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);
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
void QR_factorization_getQ(gsl_matrix *M, gsl_matrix *Q){
  int m = M->size1;
  int n = M->size2;
  int k = min(m,n);
  gsl_matrix *QR = gsl_matrix_calloc(M->size1, M->size2);
  gsl_vector *tau = gsl_vector_alloc(min(M->size1,M->size2));
  gsl_matrix_memcpy (QR, M);
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

/* invert diagonal matrix */
void invert_diagonal_matrix(gsl_matrix *Dinv, gsl_matrix *D){
  for(int i=0; i<(D->size1); i++){
    gsl_matrix_set(Dinv,i,i,1.0/(gsl_matrix_get(D,i,i)));
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

/* print matrix */
void matrix_print(gsl_matrix *M){
  for(int i=0; i<M->size1; i++){
    for(int j=0; j<M->size2; j++){
      double val = gsl_matrix_get(M, i, j);
      printf("%f  ", val);
    }
    printf("\n");
  }
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

/* copy the first k columns of M into M_out where k = M_out->ncols (M_out pre-initialized) */
void matrix_copy_first_columns(gsl_matrix *M_out, gsl_matrix *M){
  int k = M_out->size2;
  gsl_vector * col_vec;
  for(int i=0; i<k; i++){
    col_vec = gsl_vector_calloc(M->size1);
    gsl_matrix_get_col(col_vec,M,i);
    gsl_matrix_set_col(M_out,i,col_vec);
    gsl_vector_free(col_vec);
  }
}

/* append matrices side by side: C = [A, B] */
void append_matrices_horizontally(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C){
  for(int i=0; i<A->size1; i++){
    for(int j=0; j<A->size2; j++){
      gsl_matrix_set(C,i,j,gsl_matrix_get(A,i,j));
    }
  }
  for(int i=0; i<B->size1; i++){
    for(int j=0; j<B->size2; j++){
      gsl_matrix_set(C,i,A->size2 + j,gsl_matrix_get(B,i,j));
    }
  }
}

void estimate_rank_and_buildQ(gsl_matrix *M, double frac_of_max_rank, double TOL, gsl_matrix **Q, int *good_rank){
  int m = M->size1;
  int n = M->size2;
  int maxdim = round(min(m,n)*frac_of_max_rank);
  gsl_vector* vi = gsl_vector_calloc(m);
  gsl_vector* vj = gsl_vector_calloc(m);
  gsl_vector* p = gsl_vector_calloc(m);
  gsl_vector* p1 = gsl_vector_calloc(m);
  // build random matrix
  gsl_matrix* RN = gsl_matrix_calloc(n, maxdim);
  initialize_random_matrix(RN);
  // multiply to get matrix of random samples Y
  gsl_matrix* Y = gsl_matrix_calloc(m, maxdim);
  matrix_matrix_mult(M, RN, Y);
  // estimate rank k and build Q from Y
  gsl_matrix* Qbig = gsl_matrix_calloc(m, maxdim);
  gsl_matrix_memcpy(Qbig, Y);
  *good_rank = maxdim;
  int forbreak = 0;
  for(int j=0; !forbreak && j<maxdim; j++){
    gsl_matrix_get_col(vj, Qbig, j);
    for(int i=0; i<j; i++){
      gsl_matrix_get_col(vi, Qbig, i);
      project_vector(vj, vi, p);
      gsl_vector_sub(vj, p);
      if(gsl_blas_dnrm2(p) < TOL && gsl_blas_dnrm2(p1) < TOL){
        *good_rank = j;
        forbreak = 1;
        break;
      }
      gsl_vector_memcpy(p1,p);
    }
    double vec_norm = gsl_blas_dnrm2(vj);
    gsl_vector_scale(vj, 1.0/vec_norm);
    gsl_matrix_set_col(Qbig, j, vj);
  }
  gsl_matrix* Qsmall = gsl_matrix_calloc(m, *good_rank);
  *Q = gsl_matrix_calloc(m, *good_rank);
  matrix_copy_first_columns(Qsmall, Qbig);
  QR_factorization_getQ(Qsmall, *Q);

  gsl_matrix_free(RN);
  gsl_matrix_free(Y);
  gsl_matrix_free(Qbig);
  gsl_matrix_free(Qsmall);
  gsl_vector_free(p);
  gsl_vector_free(p1);
  gsl_vector_free(vi);
  gsl_vector_free(vj);
}

void estimate_rank_and_buildQ2(gsl_matrix *M, int kblock, double TOL, gsl_matrix **Y, gsl_matrix **Q, int *good_rank){
  int m = M->size1;
  int n = M->size2;
  // build random matrix
  gsl_matrix* RN = gsl_matrix_calloc(n,kblock);
  initialize_random_matrix(RN);
  // multiply to get matrix of random samples Y
  *Y = gsl_matrix_calloc(m, kblock);
  matrix_matrix_mult(M, RN, *Y);
  int ind = 0;
  int exit_loop = 0;
  while(!exit_loop){
    if(ind > 0){
      gsl_matrix_free(*Q);
    }
    *Q = gsl_matrix_calloc((*Y)->size1, (*Y)->size2);
    QR_factorization_getQ(*Y, *Q);
    // compute QtM
    gsl_matrix* QtM = gsl_matrix_calloc((*Q)->size2, M->size2);
    matrix_transpose_matrix_mult(*Q,M,QtM);
    // compute QQtM
    gsl_matrix* QQtM = gsl_matrix_calloc(M->size1, M->size2);
    matrix_matrix_mult(*Q,QtM,QQtM);
    double error_norm = 0.01*get_percent_error_between_two_mats(QQtM, M);
    *good_rank = (*Y)->size2;
    // add more samples if needed
    if(error_norm > TOL){
      gsl_matrix* Y_new = gsl_matrix_calloc(m, kblock);
      initialize_random_matrix(RN);
      matrix_matrix_mult(M, RN, Y_new);
      gsl_matrix* Y_big = gsl_matrix_calloc((*Y)->size1, (*Y)->size2 + Y_new->size2);
      append_matrices_horizontally(*Y, Y_new, Y_big);
      gsl_matrix_free(*Y);
      *Y = gsl_matrix_calloc(Y_big->size1,Y_big->size2);
      gsl_matrix_memcpy(*Y,Y_big);
      gsl_matrix_free(Y_new);
      gsl_matrix_free(Y_big);
      gsl_matrix_free(QtM);
      gsl_matrix_free(QQtM);
      ind++;
    }else{
      gsl_matrix_free(RN);
      exit_loop = 1;
    }
  }
}
