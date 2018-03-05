#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <lapacke.h>
#include <cblas.h>
#include <math.h>
#include <string.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

lapack_int LAPACKE_dgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
lapack_int LAPACKE_dormqr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );
lapack_int LAPACKE_dgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, double* a,
                           lapack_int lda, double* s, double* u, lapack_int ldu,
                           double* vt, lapack_int ldvt, double* superb );

void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);

void matrix_matrix_mult(
                        double *A, double *B, double *C,
                        int nrows_a, int ncols_a, int nrows_b, int ncols_b);

void initialize_random_matrix(double *M, int nrows, int ncols){
  time_t t;
  srand((unsigned) time(&t));
  for(int i=0; i < nrows*ncols; i++){
    //    for(int j=0; j<ncols; j++){
    M[i] = (double)rand() / (double)RAND_MAX;
      // }
  }
}

/* C = A^T*B 
 *
 * A - nrows * ncols
 * B - nrows*rank
 * C - ncols*rank
*/
void matrix_transpose_matrix_mult(
                                  double *A,
                                  double *B,
                                  double *C,
                                  int nrows_a,
                                  int ncols_a,
                                  int nrows_b,
                                  int ncols_b)
{
  cblas_dgemm(
              CblasRowMajor, CblasTrans, CblasNoTrans,
              nrows_a, ncols_b, nrows_b,
              1, A, ncols_a,
              B, ncols_b, 1,
              C, ncols_b);
}

/* compute compact QR factorization
 * Bt - input matrix. ncols * rank
 * Q - matrix in which Q is to be saved. nrows * rank
 * R - matrix in which R is to be saved. rank * rank
M is mxn; Q is mxk and R is kxk
*/
void compute_QR_compact_factorization(double *Bt, double *Q, double *R, int nrows, int ncols, int rank){
  int k = min(nrows, rank);
  double *QR_temp = (double*)malloc(sizeof(double)*nrows*rank);
  double *TAU = (double*)malloc(sizeof(double)*k);
  memcpy (QR_temp, Bt, nrows*rank);
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, nrows, rank, QR_temp, rank, TAU);
  
  for(int i=0; i<ncols; i++) {
    for(int j=0; j<rank; j++) {
      if(j>=i){
        R[i*rank + j] = QR_temp[i*rank + j];
      }
    }
  }
  
  LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'N', nrows, rank, rank, QR_temp, rank, TAU, Q, rank);
}

/* Function for calling the dgeqrf_ ATLAS routine for computing the QR decomposition.
 * The result is stored inside QR and TAU.
 * 
 * This function computes the QR decomposition and stores in QR and TAU.
 * The upper triangle (including diagonal) contains the upper trinagle matrix R.
 * QR_temp - Original input matrix. nrows*rank.
 * TAU - Temporary TAU vector used by LAPACKE. dim 1*min(nrows, rank)
 * Q - result matrix. nrows*rank
 */
// References:
// * http://www.netlib.no/netlib/lapack/double/dgeqrf.f
// * https://stackoverflow.com/a/21974753
// * https://github.com/SciRuby/nmatrix/blob/00b86992a9babef239b56a2561bfa58f1719c937/lib/nmatrix/lapacke.rb#L299
void qr_decomp(double *QR_temp, double* TAU, double* Q, int nrows, int ncols, int rank)
{
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, nrows, rank, QR_temp, rank, TAU);
  LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'N', nrows, rank, rank, QR_temp, rank, TAU, Q, rank);
}

/* compute compact QR factoriation and get Q
 * M - original matrix. dim: nrows * rank
 * Q - Q part of QR factorized matrix. Answer is returned in this vector. nrows*k. no data during input.
 * nrows - number of rows of the Matrix Q.
 * ncols - number of columns of M.
 * rank - target rank.
*/
void QR_factorization_getQ(double *M, double *Q, int nrows, int ncols, int rank){
  int k = min(nrows, rank);
  double *QR_temp = (double*)calloc(nrows*rank, sizeof(double)); // temp result from LAPACK stored in this
  double *TAU = (double*)calloc(k, sizeof(double)); // temp vector for LAPACK.
  memcpy (QR_temp, M, nrows*rank);
  for (int i = 0;i < k; ++i) QR_temp[i*k + i] = 1.0;
  
  qr_decomp (QR_temp, TAU, Q, nrows, ncols, rank);
  free(TAU);
  free(QR_temp);
}

/* build diagonal matrix from vector elements */
void build_diagonal_matrix(double *dvals, int n, double *D){
  for(int i=0; i<n; i++){
    D[i*n + i] = dvals[i];
  }
}

/* frobenius norm */
double matrix_frobenius_norm(double *M, int nrows, int ncols){
  double norm = 0;
  for(size_t i=0; i < nrows*ncols; i++){
    double val = M[i];
    norm += val*val;
  }
  
  norm = sqrt(norm);
  return norm;
}

/* P = U*S*V^T */
void form_svd_product_matrix(double *U, double *S, double *V, double *P, int nrows, int ncols, int rank)
{
  double * SVt = (double*)malloc(sizeof(double)*rank*ncols);
  matrix_matrix_mult(S, V, SVt, rank, rank, rank, ncols);
  matrix_matrix_mult(U, SVt, P, nrows, rank, rank, ncols);
}

/* calculate percent error between A and B: 100*norm(A - B)/norm(A) */
double get_percent_error_between_two_mats(double *A, double *B, int nrows, int ncols)
{
  int i;
  double *A_minus_B = (double*)malloc(sizeof(double)*nrows*ncols);
  memcpy(A_minus_B, A, nrows*ncols);
  for (i = 0; i < nrows*ncols; ++i) {
    A_minus_B[i] -= B[i];
  }
  double normA = matrix_frobenius_norm(A, nrows, ncols);
  double normA_minus_B = matrix_frobenius_norm(A_minus_B, nrows, ncols);
  return 100.0*normA_minus_B/normA;
}

/* C = A*B */
void matrix_matrix_mult(double *A, double *B, double *C, int nrows_a, int ncols_a, int nrows_b, int ncols_b){
  cblas_dgemm(
              CblasRowMajor, CblasNoTrans, CblasNoTrans,
              nrows_a, ncols_b, nrows_b,
              1, A, ncols_a, B,
              ncols_b, 1, C, ncols_b);
}

/* 
 * Outputs:
 * U -> nrows*rank
 * S -> rank*rank
 * Vt -> ncols*rank
 *
 * Inputs:
 * M -> nrows*ncols.
 */
void calculate_svd(double *U, double *S, double *Vt, double *M, int nrows, int ncols, int rank)
{
  int lda = ncols;
  int ldu = nrows;
  int ldvt = ncols;

  double superb[min(nrows, ncols) - 1];
  LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', nrows, ncols, M, lda, S, U, ldu, Vt, ldvt, superb);
}

/* computes the approximate low rank SVD of rank k of matrix M using QR method */
/* 
 * M - input matrix.
 * k - rank of the output.
 * U - U matrix from ID.
 * S - S (sigma) matrix from ID containing eigenvalues.
 * V - V matrix from ID.
 * nrows - number of rows of M.
 * ncols - number of cols of M.
 */
void randomized_low_rank_svd2(
                              double *M,
                              int rank,
                              double *U,
                              double *S,
                              double *V,
                              int nrows ,
                              int ncols)
{

  // RN = randn(n,k+p)
  // build random matrix 
  double *RN = (double*)malloc(sizeof(double)*ncols*rank);
  // calloc sets all elements to zero
  //RN = matrix_load_from_file("data/R.mtx");
  initialize_random_matrix(RN, ncols, rank);

  // Y = M * RN
  // multiply to get matrix of random samples Y
  double *Y = (double*)calloc(nrows*rank, sizeof(double)); // nrows * rank
  matrix_matrix_mult(M, RN, Y, nrows, ncols, ncols, rank);
  // [Q, R] = qr(Y,0)
  // build Q from 
  double *Q = (double*)malloc(sizeof(double)*nrows*rank);
  QR_factorization_getQ(Y, Q, nrows, ncols, rank);

  // B = Q' * A
  // form Bt = Mt*Q : nxm * mxk = nxk
  double *Bt = (double*)calloc(ncols*rank,sizeof(double));
  matrix_transpose_matrix_mult(M, Q, Bt, nrows, ncols, nrows, rank);

  double *Qhat = (double*)malloc(sizeof(double)*nrows*rank);
  double *Rhat = (double*)malloc(sizeof(double)*rank*rank);
  // Bt -> ncols * rank, Qhat -> nrows * rank, Rhat -> rank, rank
  compute_QR_compact_factorization(Bt, Qhat, Rhat, nrows, ncols, rank);

  // [Uhat, D, V] = svd(B,'econ')
  // compute SVD of Rhat (kxk)
  double *Uhat = (double*)malloc(sizeof(double)*rank*rank);
  double *Sigmahat = (double*)malloc(sizeof(double)*rank*rank);
  double *Vhat = (double*)malloc(sizeof(double)*rank*rank);
  memcpy(Uhat, Rhat, rank*rank);
  calculate_svd(Uhat, Sigmahat, Vhat, Bt, rank, rank, rank);
  
  // U = Q * Uhat
  // record singul0ar values
  build_diagonal_matrix(Sigmahat, rank, S);
  // U = Q*Vhat
  matrix_matrix_mult(Q, Vhat, U, nrows, rank, rank, rank);
  // V = Qhat*Uhat
  matrix_matrix_mult(Qhat, Uhat, V, nrows, rank, rank, rank);
  // free stuff
  free(RN);
  free(Y);
  free(Q);
  free(Rhat);
  free(Qhat);
  free(Uhat);
  free(Vhat);
  free(Bt);
}
